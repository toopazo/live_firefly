#!/usr/bin/env python


from __future__ import print_function

import datetime
# import select
# import time
import time
from timeit import default_timer as timer

import threading
import queue
from enum import Enum

try:
    from pymavlink import mavutil
    import serial
except ImportError:
    print("Failed to import pymavlink.")
    print("You may need to install it with 'pip install pymavlink pyserial'")
    print("")
    raise
from argparse import ArgumentParser


class MavlinkSerialPort:
    """
    An object that looks like a serial port, but
    transmits using mavlink SERIAL_CONTROL packets
    """

    def __init__(self, portname, baudrate, devnum=0, debug=0):
        self.baudrate = 0
        self._debug = debug
        self.buf = ''
        self.port = devnum
        self.debug("Connecting with MAVLink to %s ..." % portname)
        self.mav = mavutil.mavlink_connection(portname, autoreconnect=True, baud=baudrate)
        self.mav.wait_heartbeat()
        self.debug("HEARTBEAT OK\n")
        self.debug("Locked serial device\n")

    def debug(self, s, level=1):
        """write some debug text"""
        if self._debug >= level:
            print(s)

    def write(self, b):
        """write some bytes"""
        self.debug("sending '%s' (0x%02x) of len %u\n" % (b, ord(b[0]), len(b)), 2)
        while len(b) > 0:
            n = len(b)
            if n > 70:
                n = 70
            buf = [ord(x) for x in b[:n]]
            buf.extend([0]*(70-len(buf)))
            self.mav.mav.serial_control_send(
                self.port,
                mavutil.mavlink.SERIAL_CONTROL_FLAG_EXCLUSIVE |
                mavutil.mavlink.SERIAL_CONTROL_FLAG_RESPOND,
                0,
                0,
                n,
                buf
            )
            b = b[n:]

    def close(self):
        self.mav.mav.serial_control_send(self.port, 0, 0, 0, 0, [0]*70)

    def _recv(self):
        """read some bytes into self.buf"""
        m = self.mav.recv_match(condition='SERIAL_CONTROL.count!=0',
                                type='SERIAL_CONTROL', blocking=True,
                                timeout=0.03)
        if m is not None:
            if self._debug > 2:
                print(m)
            data = m.data[:m.count]
            self.buf += ''.join(str(chr(x)) for x in data)

    def read(self, n):
        """read some bytes"""
        if len(self.buf) == 0:
            self._recv()
        if len(self.buf) > 0:
            if n > len(self.buf):
                n = len(self.buf)
            ret = self.buf[:n]
            self.buf = self.buf[n:]
            if self._debug >= 2:
                for b in ret:
                    self.debug("read 0x%x" % ord(b), 2)
            return ret
        return ''


class FireflyMavEnum(Enum):
    stop_running = 1
    nsh_command = 2


class FireflyMavMsg:
    def __init__(self, key, val):
        assert isinstance(key, FireflyMavEnum)
        self.key = key
        self.val = val


class FireflyMavlink:
    time_prev = datetime.datetime.now()

    @staticmethod
    def check_timeout(timeout):
        time_now = datetime.datetime.now()
        delta = time_now - FireflyMavlink.time_prev
        if delta.seconds >= timeout:
            FireflyMavlink.time_prev = time_now
            return True
        else:
            return False

    def __init__(self, port, baudrate):
        self.port = port
        self.baudrate = baudrate
        self.cmd_rate = 3

    def start(self, _queue):
        _thread = threading.Thread(target=self.run, args=(_queue,))
        _thread.start()
        return _thread

    def run(self, _queue):
        print(f"[{FireflyMavlink.__name__}] Connecting to MAVLINK at {self.port} {self.baudrate} ..")
        mav_serial = MavlinkSerialPort(self.port, self.baudrate, devnum=10)
        print(f"[{FireflyMavlink.__name__}] Connected to MAVLINK at {self.port} {self.baudrate} ..")

        # make sure the shell is started
        mav_serial.write('\n')

        try:
            next_heartbeat_time = timer()

            while True:
                try:
                    # fm_msg = _queue.get(block=False)
                    fm_msg = _queue.get(block=True, timeout=0.1)
                    assert isinstance(fm_msg, FireflyMavMsg)
                    print(f'A {FireflyMavMsg.__name__} was received')
                    if fm_msg.key == FireflyMavEnum.stop_running and fm_msg.val:
                        time.sleep(self.cmd_rate)
                        mav_serial.close()
                        return
                    if fm_msg.key == FireflyMavEnum.nsh_command:
                        cmd = fm_msg.val
                        mav_serial.write(cmd + '\n')
                        print(cmd)
                except queue.Empty:
                    pass

                data = mav_serial.read(4096*4)
                if data and len(data) > 0:
                    print(data, end='')

                # handle heartbeat sending
                heartbeat_time = timer()
                if heartbeat_time > next_heartbeat_time:
                    mav_serial.mav.mav.heartbeat_send(
                        mavutil.mavlink.MAV_TYPE_GCS,
                        mavutil.mavlink.MAV_AUTOPILOT_GENERIC, 0, 0, 0
                    )
                    next_heartbeat_time = heartbeat_time + 1

        except serial.serialutil.SerialException as e:
            print(e)
        except KeyboardInterrupt:
            mav_serial.close()


if __name__ == '__main__':
    fm = FireflyMavlink(port='/dev/ttyACM1', baudrate=57600)

    fm_queue = queue.Queue()
    fm_thread = fm.start(fm_queue)

    cmd_rate = 3
    for i in range(0, 3):
        if FireflyMavlink.check_timeout(timeout=cmd_rate):
            nsh_cmd = f'firefly write_delta {i} {i} 1'
            fm_queue.put(FireflyMavMsg(FireflyMavEnum.nsh_command, nsh_cmd))
        time.sleep(cmd_rate)
    fm_queue.put(FireflyMavMsg(FireflyMavEnum.stop_running, True))

    print('Calling join ..')
    fm_thread.join(timeout=60 * 1)

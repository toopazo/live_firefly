#!/usr/bin/env python


from __future__ import print_function

import datetime
import sys
# import select
import termios
# import time
import time
from timeit import default_timer as timer
import threading

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


class FireflyMavCmd:
    time_prev = datetime.datetime.now()
    mavcmd_cnt = 0
    mavcmd_array = [
        'firefly write_delta +0.0 +0.0 1',  # zero delta
        'firefly write_delta +0.0 +0.0 1',  # zero delta
        'firefly write_delta +0.0 +0.0 1',  # zero delta
        'firefly write_delta +0.1 +0.1 1',
        'firefly write_delta +0.2 +0.2 1',
        'firefly write_delta +0.3 +0.3 1',
        'firefly write_delta +0.4 +0.4 1',
        'firefly write_delta +0.5 +0.5 1',
        'firefly write_delta +0.6 +0.6 1',
        'firefly write_delta +0.7 +0.7 1',
        'firefly write_delta +0.8 +0.8 1',
        'firefly write_delta +0.9 +0.9 1',
        'firefly write_delta +1.0 +1.0 1',  # highest +delta
        'firefly write_delta +0.8 +0.8 1',  # coming back to zero delta
        'firefly write_delta +0.6 +0.6 1',  # coming back to zero delta
        'firefly write_delta +0.4 +0.4 1',  # coming back to zero delta
        'firefly write_delta +0.2 +0.2 1',  # coming back to zero delta
        # 'firefly write_delta -0.1 -0.1 1',
        # 'firefly write_delta -0.2 -0.2 1',
        # 'firefly write_delta -0.3 -0.3 1',
        # 'firefly write_delta -0.4 -0.4 1',
        # 'firefly write_delta -0.5 -0.5 1',
        # 'firefly write_delta -0.6 -0.6 1',
        # 'firefly write_delta -0.7 -0.7 1',
        # 'firefly write_delta -0.8 -0.8 1',
        # 'firefly write_delta -0.9 -0.9 1',
        # 'firefly write_delta -1.0 -1.0 1',  # highest -delta
        # 'firefly write_delta -0.8 -0.8 1',  # coming back to zero delta
        # 'firefly write_delta -0.6 -0.6 1',  # coming back to zero delta
        # 'firefly write_delta -0.4 -0.4 1',  # coming back to zero delta
        # 'firefly write_delta -0.2 -0.2 1',  # coming back to zero delta
    ]
    mavcmd_iterator = iter(mavcmd_array)

    @staticmethod
    def check_timeout():
        time_now = datetime.datetime.now()
        delta = time_now - FireflyMavCmd.time_prev
        if delta.seconds >= 10:
            FireflyMavCmd.time_prev = time_now
            return True
        else:
            return False

    @staticmethod
    def next_mavcmd():
        try:
            mavcmd = next(FireflyMavCmd.mavcmd_iterator)
        except StopIteration:
            mavcmd = 'firefly write_delta 0 0 0'
        return mavcmd


class FireflyMavshell:
    def __init__(self, port, baudrate):
        self.port = port
        self.baudrate = baudrate
        self.keep_running = True

    def run(self):
        print(f"[FireflyMavshell] Connecting to MAVLINK at {self.port} {self.baudrate} ..")
        mav_serial = MavlinkSerialPort(self.port, self.baudrate, devnum=10)

        # make sure the shell is started
        mav_serial.write('\n')

        # setup the console, so we can read one char at a time
        fd_in = sys.stdin.fileno()
        old_attr = termios.tcgetattr(fd_in)
        new_attr = termios.tcgetattr(fd_in)
        new_attr[3] = new_attr[3] & ~termios.ECHO  # lflags
        new_attr[3] = new_attr[3] & ~termios.ICANON

        try:
            termios.tcsetattr(fd_in, termios.TCSANOW, new_attr)
            cur_line = ''
            command_history = []
            cur_history_index = 0

            def erase_last_n_chars(N):
                if N == 0: return
                CURSOR_BACK_N = '\x1b[' + str(N) + 'D'
                ERASE_END_LINE = '\x1b[K'
                sys.stdout.write(CURSOR_BACK_N + ERASE_END_LINE)

            next_heartbeat_time = timer()

            cnt = 0
            while self.keep_running:
                cnt = cnt + 1
                if FireflyMavCmd.check_timeout():
                    mavcmd = FireflyMavCmd.next_mavcmd()
                    mav_serial.write(mavcmd + '\n')

                data = mav_serial.read(4096)
                if data and len(data) > 0:
                    sys.stdout.write(data)
                    sys.stdout.flush()

                # handle heartbeat sending
                heartbeat_time = timer()
                if heartbeat_time > next_heartbeat_time:
                    mav_serial.mav.mav.heartbeat_send(
                        mavutil.mavlink.MAV_TYPE_GCS,
                        mavutil.mavlink.MAV_AUTOPILOT_GENERIC, 0, 0, 0
                    )
                    next_heartbeat_time = heartbeat_time + 1
            mav_serial.close()

        except serial.serialutil.SerialException as e:
            print(e)
        except KeyboardInterrupt:
            mav_serial.close()
        finally:
            termios.tcsetattr(fd_in, termios.TCSADRAIN, old_attr)

    def close(self):
        self.keep_running = False


if __name__ == '__main__':
    fm = FireflyMavshell(port='/dev/ttyACM1', baudrate=57600)
    fm_thr = threading.Thread(target=fm.run())
    fm_thr.start()
    time.sleep(30)
    fm.close()
    print('waiting to close ..')
    fm_thr.join(timeout=60*1)
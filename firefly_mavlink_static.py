#!/usr/bin/env python

"""
Open a shell over MAVLink.

@author: Beat Kueng (beat-kueng@gmx.net)
"""


from __future__ import print_function

import datetime
import sys
import select
import termios
import time
from timeit import default_timer as timer

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
        'firefly write_delta +0.5 +0.5 1',  # highest +delta
        'firefly write_delta +0.4 +0.4 1',
        'firefly write_delta +0.3 +0.3 1',
        'firefly write_delta +0.2 +0.2 1',
        'firefly write_delta +0.1 +0.1 1',
        'firefly write_delta +0.0 +0.0 1',  # zero delta
        'firefly write_delta -0.1 -0.1 1',
        'firefly write_delta -0.2 -0.2 1',
        'firefly write_delta -0.3 -0.3 1',
        'firefly write_delta -0.4 -0.4 1',
        'firefly write_delta -0.5 -0.5 1',  # lowest +delta
        'firefly write_delta -0.4 -0.4 1',
        'firefly write_delta -0.3 -0.3 1',
        'firefly write_delta -0.2 -0.2 1',
        'firefly write_delta -0.1 -0.1 1',
        'firefly write_delta +0.0 +0.0 1',  # zero delta
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


def main():
    parser = ArgumentParser(description=__doc__)
    help_msg = 'Mavlink port name: serial: DEVICE[,BAUD], udp: IP:PORT, ' \
               'tcp: tcp:IP:PORT. Eg: /dev/ttyUSB0 or 0.0.0.0:14550. ' \
               'Auto-detect serial if not given.'
    parser.add_argument(
        'port', metavar='PORT', nargs='?', default = None,
        help=help_msg)
    parser.add_argument(
        "--baudrate", "-b", dest="baudrate", type=int,
        help="Mavlink port baud rate (default=57600)", default=57600)
    args = parser.parse_args()

    if args.port is None:
        if sys.platform == "darwin":
            args.port = "/dev/tty.usbmodem1"
        else:
            serial_list = mavutil.auto_detect_serial(
                preferred_list=[
                    '*FTDI*', "*Arduino_Mega_2560*", "*3D_Robotics*",
                    "*USB_to_UART*", '*PX4*', '*FMU*', "*Gumstix*"]
            )

            if len(serial_list) == 0:
                print("Error: no serial connection found")
                return

            if len(serial_list) > 1:
                print('Auto-detected serial ports are:')
                for port in serial_list:
                    print(" {:}".format(port))
            print('Using port {:}'.format(serial_list[0]))
            args.port = serial_list[0].device

    print(f"Connecting to MAVLINK at {args.port} {args.baudrate} ..")
    mav_serialport = MavlinkSerialPort(args.port, args.baudrate, devnum=10)

    # make sure the shell is started
    mav_serialport.write('\n')

    # setup the console, so we can read one char at a time
    fd_in = sys.stdin.fileno()
    old_attr = termios.tcgetattr(fd_in)
    new_attr = termios.tcgetattr(fd_in)
    new_attr[3] = new_attr[3] & ~termios.ECHO # lflags
    new_attr[3] = new_attr[3] & ~termios.ICANON

    try:
        termios.tcsetattr(fd_in, termios.TCSANOW, new_attr)
        cur_line = ''
        command_history = []
        cur_history_index = 0

        def erase_last_n_chars(N):
            if N == 0: return
            CURSOR_BACK_N = '\x1b['+str(N)+'D'
            ERASE_END_LINE = '\x1b[K'
            sys.stdout.write(CURSOR_BACK_N + ERASE_END_LINE)

        next_heartbeat_time = timer()

        cnt = 0
        while True:
            cnt = cnt + 1
            if FireflyMavCmd.check_timeout():
                mavcmd = FireflyMavCmd.next_mavcmd()
                mav_serialport.write(mavcmd+'\n')

            data = mav_serialport.read(4096)
            if data and len(data) > 0:
                sys.stdout.write(data)
                sys.stdout.flush()

            # handle heartbeat sending
            heartbeat_time = timer()
            if heartbeat_time > next_heartbeat_time:
                mav_serialport.mav.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_GCS,
                    mavutil.mavlink.MAV_AUTOPILOT_GENERIC, 0, 0, 0
                )
                next_heartbeat_time = heartbeat_time + 1

    except serial.serialutil.SerialException as e:
        print(e)

    except KeyboardInterrupt:
        mav_serialport.close()

    finally:
        termios.tcsetattr(fd_in, termios.TCSADRAIN, old_attr)


if __name__ == '__main__':
    main()


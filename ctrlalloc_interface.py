#!/usr/bin/env python

"""
Open a shell over MAVLink.

@author: Beat Kueng (beat-kueng@gmx.net)
"""


from __future__ import print_function
import sys
import select
import termios
import time
from timeit import default_timer as timer

import numpy as np

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


class CtrlAllocIface:
    def __init__(self, port):
        print("Connecting to MAVLINK...")
        mav_serialport = MavlinkSerialPort(port, 57600, devnum=10)
        self.mav_serialport = mav_serialport

    def send_delta_commands(self, delta):
        delta_str = [str(d) for d in delta]
        vals = " ".join(delta_str)
        cmd = f'toopazo {vals}'
        # print(cmd)
        self.mav_serialport.write(cmd)
        time.sleep(1)
        res = self.mav_serialport.read(4096*10)
        return res


if __name__ == '__main__':
    uval = float(sys.argv[1])
    ctrlalloc_iface = CtrlAllocIface(port='/dev/ttyUSB0')
    udelta = np.ones(8) * uval
    ures = ctrlalloc_iface.send_delta_commands(udelta)
    print(f'ctrlalloc_iface.send_delta_commands returned {ures}')

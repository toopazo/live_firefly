from pymavlink import mavutil
import numpy as np


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
        self.mav.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GENERIC, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
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


def check_nsh_input(cmd_input, old_cmd):
    check_flag = np.zeros(3)

    try:
        cmd = [float(a) for a in cmd_input.split(' ')]
        check_flag[0] = (len(cmd) == 2)
        check_flag[1] = np.around(np.abs(old_cmd[0] - cmd[0]), decimals=2) <= 0.2
        check_flag[2] = np.around(np.abs(old_cmd[1] - cmd[1]), decimals=2) <= 0.2

    except ValueError:  # Raised if input is not float
        return -1, False, "Commands could not be casted into float values!"

    except IndexError:  # Raised it there are less than 2 commands
        return -1, False, "Less than 2 commands. Please enter exact 2 commands!"

    # check if there are exactly two commands
    if not check_flag[0]:
        return -1, False, "More than 2 commands. Please enter exact two commands!"

    # check if delta between commands is lower than 0.2
    if not check_flag[1] or not check_flag[2]:
        return -1, False, f'Difference to previous commands ({old_cmd[0]}, {old_cmd[1]}) to large. Please limit steps to a maximum delta of 0.2'

    # final check if all flags are true
    if np.all(check_flag == True):
        return cmd, True, "Arguments valid!"
    else:
        return -1, False, "Unknown Error!"
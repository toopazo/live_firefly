#!/usr/bin/python3
import datetime
import sys
# import numpy as np
# import signal
# import time
# import datetime
import time
import queue

from toopazo_tools.file_folder import FileFolderTools as FFTools
from toopazo_tools.telemetry import TelemetryLogger
from firefly_mavlink_live import FireflyMavEnum, FireflyMavMsg, FireflyMavlink

from live_ars.ars_interface import ArsIface
from live_esc.kde_uas85uvc.kdecan_interface import KdeCanIface
from ctrlalloc_interface import CtrlAllocIface


class EscIfaceWrapper:
    def __init__(self):
        self.esc = KdeCanIface()
        self.esc_arr = list(range(11, 19))

    def get_data(self):
        # get data
        resp_arr = self.esc.get_data_esc_arr(self.esc_arr)

        # for each targetid, get data
        log_data_final = ""
        for resp in resp_arr:
            [telap, targetid,
             voltage, current, rpm, temp, warning,
             inthrottle, outthrottle] = resp

            log_data = "%s, %s, %04.2f, %s, %07.2f, %s, %s, %s, %s " % \
                       (telap, targetid,
                        voltage, current, rpm, temp, warning,
                        inthrottle, outthrottle)
            if len(log_data_final) == 0:
                log_data_final = log_data
            else:
                log_data_final = log_data_final + "\r\n" + log_data
            # self.log_fd.write(log_data + "\r\n")
        return log_data_final

    def close(self):
        self.esc.kdecan_bus.shutdown()


class ArsIfaceWrapper:
    def __init__(self, port):
        self.ars = ArsIface(port)

    def get_data(self):
        log_data = self.ars.safely_read_data()
        return log_data

    def close(self):
        self.ars.serial_thread_close()


class FireflyIfaceWrapper:
    def __init__(self, ars_port, ctrlalloc_port):
        self.ars = ArsIfaceWrapper(ars_port)
        self.esc = EscIfaceWrapper()
        self.ctrlalloc = CtrlAllocIface(ctrlalloc_port)
        self.datetime_0 = datetime.datetime.now()
        # self.delta_arr = delta_arr

    def get_data(self):
        ars_data = self.ars.get_data()
        esc_data = self.esc.get_data()
        log_data = f"{ars_data}, {esc_data}"
        return log_data

    def close(self):
        self.ars.close()
        # self.esc.close()

    @staticmethod
    def get_header():
        fields = ["sps", "mills", "secs", "dtmills",
                  "cur1", "cur2", "cur3", "cur4",
                  "cur5", "cur6", "cur7", "cur8",
                  "rpm1", "rpm2", "rpm3", "rpm4",
                  "rpm5", "rpm6", "rpm7", "rpm8"]
        ars_header = ", ".join(fields)
        esc_header = "time s, escid, " \
                     "voltage V, current A, angVel rpm, temp degC, warning, " \
                     "inthtl us, outthtl perc"
        firefly_header = f"{ars_header}, {esc_header}"
        return firefly_header


def parse_user_arg(folder):
    folder = FFTools.full_path(folder)
    print('target folder {}'.format(folder))
    if FFTools.is_folder(folder):
        cwd = FFTools.get_cwd()
        print('current folder {}'.format(cwd))
    else:
        arg = '{} is not a folder'.format(folder)
        raise RuntimeError(arg)
    return folder


def test_mavlink_shell():
    fm = FireflyMavlink(port='/dev/ttyACM1', baudrate=57600)

    fm_queue = queue.Queue()
    fm_thread = fm.start(fm_queue)

    cmd_rate = 3
    for i in range(0, 3):
        # if FireflyMavlink.check_timeout(timeout=cmd_rate):
        nsh_cmd = f'firefly write_delta {i} {i} 1'
        fm_queue.put(FireflyMavMsg(FireflyMavEnum.nsh_command, nsh_cmd))
        time.sleep(cmd_rate)
    fm_queue.put(FireflyMavMsg(FireflyMavEnum.stop_running, True))

    print('Calling join ..')
    fm_thread.join(timeout=60 * 1)


def test_fireflyiface():
    firefly_iface = FireflyIfaceWrapper(
        ars_port='/dev/ttyACM0', ctrlalloc_port='/dev/ttyUSB0')
    time0 = datetime.datetime.now()

    _sampling_period = 1
    while True:
        print(FireflyIfaceWrapper.get_header())
        log_data = firefly_iface.get_data()
        print(log_data)
        TelemetryLogger.busy_waiting(
            time0, _sampling_period, _sampling_period / 8)


if __name__ == '__main__':
    test_fireflyiface()

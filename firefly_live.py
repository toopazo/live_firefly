#!/usr/bin/python3

# import numpy as np
# import signal
# import time
# import datetime
import time
import queue
import numpy as np

from toopazo_tools.file_folder import FileFolderTools as FFTools
from toopazo_tools.telemetry import TelemetryLogger
from firefly_mavlink_live import FireflyMavEnum, FireflyMavMsg, FireflyMavlink
from firefly_optimizer import FireflyOptimizer

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
                # log_data_final = log_data_final + "\r\n" + log_data
                log_data_final = log_data_final + ", " + log_data
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


class SensorIfaceWrapper:
    def __init__(self, ars_port):
        self.ars = ArsIfaceWrapper(ars_port)
        self.esc = EscIfaceWrapper()
        # self.ctrlalloc = CtrlAllocIface(ctrlalloc_port)
        # self.datetime_0 = datetime.datetime.now()
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
        fields = [
            "sps", "mills", "secs", "dtmills",
            "rpm1", "rpm2", "rpm3", "rpm4",
            "rpm5", "rpm6", "rpm7", "rpm8",
            "cur1", "cur2", "cur3", "cur4",
            "cur5", "cur6", "cur7", "cur8"
        ]
        ars_header = ", ".join(fields)
        esc_header = "time s, escid, " \
                     "voltage V, current A, angVel rpm, temp degC, warning, " \
                     "inthtl us, outthtl perc"
        sensor_header = f"{ars_header}, {esc_header}"
        return sensor_header


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

    cmd_period = 3
    for i in range(0, 3):
        # if FireflyMavlink.check_timeout(timeout=cmd_rate):
        nsh_cmd = f'firefly write_delta {i} {i} 1'
        fm_queue.put(FireflyMavMsg(FireflyMavEnum.nsh_command, nsh_cmd))
        time.sleep(cmd_period)
    fm_queue.put(FireflyMavMsg(FireflyMavEnum.stop_running, True))

    print('Calling join ..')
    fm_thread.join(timeout=60 * 1)


def test_sensor_iface():
    sensor_iface = SensorIfaceWrapper(ars_port='/dev/ttyACM0')
    # time0 = datetime.datetime.now()
    time0 = time.time()

    log_folder = parse_user_arg('/home/pi/live_firefly/logs')
    log_ext = ".firefly"
    telem_logger = TelemetryLogger(log_folder, sensor_iface, log_ext)

    sampling_period = 3
    while True:
        print(SensorIfaceWrapper.get_header())
        log_data = sensor_iface.get_data()
        print(f'log_data {log_data}')
        fcost = FireflyOptimizer.sensor_data_to_cost_fnct(sensor_data=log_data)
        print(f'fcost {fcost}')
        TelemetryLogger.busy_waiting(time0, sampling_period, sampling_period / 8)
        telem_logger.save_data(log_data=f'{log_data}, {fcost}', log_header='')


def test_optimizer():
    sensor_iface = SensorIfaceWrapper(ars_port='/dev/ttyACM0')
    # time0 = datetime.datetime.now()
    time0 = time.time()

    log_folder = parse_user_arg('/home/pi/live_firefly/logs')
    log_ext = ".firefly"
    telem_logger = TelemetryLogger(log_folder, sensor_iface, log_ext)

    fm = FireflyMavlink(port='/dev/ttyACM1', baudrate=57600)
    fm_queue = queue.Queue()
    fm_thread = fm.start(fm_queue)

    cmd_period = 3
    sampling_period = 0.3
    cost_m38_arr = []
    cost_m47_arr = []
    avg_cost_m38 = 0
    avg_cost_m47 = 0
    num_samples = np.ceil(cmd_period/sampling_period)
    nsh_delta = 0
    nsh_delta_prev = 0
    cnt = 0
    try:
        while True:
            cnt = cnt + 1

            log_data = sensor_iface.get_data()
            fcost = FireflyOptimizer.sensor_data_to_cost_fnct(sensor_data=log_data)
            print(f'fcost {[round(e, 4) for e in fcost]}')
            TelemetryLogger.busy_waiting(time0, sampling_period, sampling_period / 8)

            cost_m38 = fcost[3 - 1] + fcost[8 - 1]
            cost_m47 = fcost[4 - 1] + fcost[7 - 1]
            cost_m38_arr.append(cost_m38)
            cost_m47_arr.append(cost_m47)

            if cnt >= num_samples:
                k = 0.1
                avg_cost_m38 = np.average(cost_m38_arr)
                avg_cost_m47 = np.average(cost_m47_arr)
                nsh_delta = nsh_delta - k * (avg_cost_m38 + avg_cost_m47)
                if nsh_delta >= +0.5:
                    nsh_delta = +0.5
                if nsh_delta <= -0.5:
                    nsh_delta = -0.5
                max_delta_change = 0.1
                if (nsh_delta - nsh_delta_prev) >= +max_delta_change:
                    nsh_delta = nsh_delta_prev + max_delta_change
                if (nsh_delta - nsh_delta_prev) <= -max_delta_change:
                    nsh_delta = nsh_delta_prev - max_delta_change
                nsh_delta_prev = nsh_delta

                nsh_cmd = f'firefly write_delta {nsh_delta} {nsh_delta} 1'
                fm_queue.put(FireflyMavMsg(FireflyMavEnum.nsh_command, nsh_cmd))

                cost_m38_arr = []
                cost_m47_arr = []

            optim_data = f'{nsh_delta}, {avg_cost_m38}, {avg_cost_m47}'
            log_data = f'{log_data}, {fcost}, {optim_data}'
            telem_logger.save_data(log_data=log_data, log_header='')

    except KeyboardInterrupt:
        fm_queue.put(FireflyMavMsg(FireflyMavEnum.stop_running, True))
        fm_thread.join(timeout=60 * 1)


if __name__ == '__main__':
    # test_mavlink_shell()
    # test_sensor_iface()
    test_optimizer()

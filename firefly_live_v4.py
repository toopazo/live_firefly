
import time
import queue
import numpy as np

from toopazo_tools.file_folder import FileFolderTools as FFTools
from toopazo_tools.telemetry import TelemetryLogger
from firefly_mavlink_live import FireflyMavEnum, FireflyMavMsg, FireflyMavlink

from live_ars.ars_interface import ArsIface
from live_esc.kde_uas85uvc.kdecan_interface import KdeCanIface
# from ctrlalloc_interface import CtrlAllocIface


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


class EscOptimizer:
    @staticmethod
    def parse_sensor_data(sensor_data):
        # fields = [
        #     "sps", "mills", "secs", "dtmills",
        #     "rpm1", "rpm2", "rpm3", "rpm4",
        #     "rpm5", "rpm6", "rpm7", "rpm8",
        #     "cur1", "cur2", "cur3", "cur4",
        #     "cur5", "cur6", "cur7", "cur8"
        # ]
        # ars_header = ", ".join(fields)
        # esc_header = "time s, escid, " \
        #              "voltage V, current A, angVel rpm, temp degC, warning" \
        #              ", inthtl us, outthtl perc"
        # sensor_header = f"{ars_header}, {esc_header}"
        parsed_data_arr = []
        for e in sensor_data.split(','):
            try:
                val = float(e.strip())
            except ValueError:
                val = e.strip()
            parsed_data_arr.append(val)

        parsed_data_dict = {
            'sps': parsed_data_arr[0],
            'mills': parsed_data_arr[1],
            'secs': parsed_data_arr[2],
            'dtmills': parsed_data_arr[3],
            'rpm1': parsed_data_arr[4],
            'rpm2': parsed_data_arr[5],
            'rpm3': parsed_data_arr[6],
            'rpm4': parsed_data_arr[7],
            'rpm5': parsed_data_arr[8],
            'rpm6': parsed_data_arr[9],
            'rpm7': parsed_data_arr[10],
            'rpm8': parsed_data_arr[11],
            'cur1': parsed_data_arr[12],
            'cur2': parsed_data_arr[13],
            'cur3': parsed_data_arr[14],
            'cur4': parsed_data_arr[15],
            'cur5': parsed_data_arr[16],
            'cur6': parsed_data_arr[17],
            'cur7': parsed_data_arr[18],
            'cur8': parsed_data_arr[19],
            # adding reordered data
            'rpm_13': parsed_data_arr[4],  # rmp1
            'rpm_18': parsed_data_arr[5],  # rmp2
            'rpm_14': parsed_data_arr[6],  # rmp3
            'rpm_17': parsed_data_arr[7],  # rmp4
            'cur_13': parsed_data_arr[12],  # cur1
            'cur_18': parsed_data_arr[13],  # cur2
            'cur_14': parsed_data_arr[14],  # cur3
            'cur_17': parsed_data_arr[15],  # cur4
        }
        # Dec9 19:39

        # pwm test -c 1 -p 1000
        # angVel_11     OK
        # current_11    channel is dead
        # voltage_11    OK

        # pwm test -c 2 -p 1000
        # angVel_12     OK
        # current_12    channel is dead
        # voltage_12    OK

        # pwm test -c 3 -p 1000
        # angVel_13     OK
        # current_13    channel is dead     corresponds to cur1
        # voltage_13    OK

        # pwm test -c 4 -p 1000
        # angVel_14     OK
        # current_14    channel is dead     corresponds to cur3
        # voltage_14    OK

        # pwm test -c 5 -p 1000
        # angVel_15     OK
        # current_15    channel is dead
        # voltage_15    OK

        # pwm test -c 6 -p 1000
        # angVel_16     OK
        # current_16    channel is dead
        # voltage_16    OK

        # pwm test -c 7 -p 1000
        # angVel_17     OK
        # current_17    channel is dead     corresponds to cur4
        # voltage_17    OK

        # pwm test -c 8 -p 1000
        # angVel_18     OK
        # current_18    channel is dead     corresponds to cur2
        # voltage_18    OK

        # cur1 corresponds to motor3
        # cur2 corresponds to motor8
        # cur3 corresponds to motor4
        # cur4 corresponds to motor7

        num_rotors = 8
        for i in range(0, num_rotors):
            indx = 20 + 9 * i
            escid = str(10 + i + 1)
            # print(f'indx {indx} escid {escid}')

            try:
                parsed_data_dict[f'time_{escid}'] = parsed_data_arr[
                    indx + 0]  # 20 + 9*0 = 20, # 20 + 9*1 = 29
                parsed_data_dict[f'escid_{escid}'] = parsed_data_arr[indx + 1]
                parsed_data_dict[f'voltage_{escid}'] = parsed_data_arr[indx + 2]
                parsed_data_dict[f'current_{escid}'] = parsed_data_arr[indx + 3]
                parsed_data_dict[f'angVel_{escid}'] = parsed_data_arr[indx + 4]
                parsed_data_dict[f'temp_{escid}'] = parsed_data_arr[indx + 5]
                parsed_data_dict[f'warning_{escid}'] = parsed_data_arr[indx + 6]
                parsed_data_dict[f'inthtl_{escid}'] = parsed_data_arr[indx + 7]
                parsed_data_dict[f'outthtl_{escid}'] = parsed_data_arr[indx + 8]
            except IndexError:
                pass
        return parsed_data_dict

    @staticmethod
    def sensor_data_to_cost_fnct(sensor_data):
        parsed_data = EscOptimizer.parse_sensor_data(sensor_data)
        # print(f'parsed_data {parsed_data}')
        try:
            # pprint.pprint(FireflyOptimizer.filter_parsed_data(parsed_data))
            pass
        except KeyError:
            print('[sensor_data_to_cost_fnct] error parsing data')

        cost_arr = []
        num_rotors = 8
        for i in range(0, num_rotors):
            escid = str(10 + i + 1)
            try:
                # esc data
                cost = parsed_data[f'voltage_{escid}'] * parsed_data[
                    f'current_{escid}']

                # ars data
                # cost = parsed_data[f'voltage_{escid}'] * parsed_data[
                #     f'cur_{escid}']
            except KeyError:
                cost = -1
            cost_arr.append(cost)
        return cost_arr

    @staticmethod
    def run_optimizer():
        sensor_iface = SensorIfaceWrapper(ars_port='/dev/ttyACM0')
        # time0 = datetime.datetime.now()
        time0 = time.time()

        log_folder = parse_user_arg('/home/pi/live_firefly/logs')
        log_ext = ".firefly"
        telem_logger = TelemetryLogger(log_folder, sensor_iface, log_ext)

        fmavl = FireflyMavlink(port='/dev/ttyACM1', baudrate=57600)
        fmavl_queue = queue.Queue()
        fmavl_thread = fmavl.start(fmavl_queue)

        cmd_period = 6
        # sampling_period = 0.3
        sampling_period = 0.1
        cost_m38_arr = []
        cost_m47_arr = []
        avg_cost_m38 = 0
        avg_cost_m47 = 0
        avg_cost_tot = 0
        num_samples = np.ceil(cmd_period/sampling_period)
        cnt_samples = 0
        nsh_delta = 0
        nsh_delta_prev = 0
        max_abs_rate = 0.05
        avg_cost_tot_up = 0
        avg_cost_tot_down = 0
        avg_cost_tot_prev = 0
        optimizer_state = 0
        try:
            while True:
                cnt_samples = cnt_samples + 1

                sensor_data = sensor_iface.get_data()
                fcost = EscOptimizer.sensor_data_to_cost_fnct(
                    sensor_data=sensor_data)
                # print(f'fcost {[round(e, 4) for e in fcost]}')
                TelemetryLogger.busy_waiting(
                    time0, sampling_period, sampling_period / 8)

                cost_m38 = fcost[3 - 1] + fcost[8 - 1]
                cost_m47 = fcost[4 - 1] + fcost[7 - 1]
                cost_m38_arr.append(cost_m38)
                cost_m47_arr.append(cost_m47)

                # Low battery check
                if cnt_samples >= int(num_samples/2):
                    parsed_data = EscOptimizer.parse_sensor_data(sensor_data)
                    m13_volage = parsed_data[f'voltage_13']
                    min_voltage = (3.4*4)*2
                    if m13_volage <= min_voltage:
                        print(f'Warning, low battery {m13_volage} !!!!!!!!!!!')

                # Processing of previous time window
                if cnt_samples >= num_samples:
                    avg_cost_m38 = np.average(cost_m38_arr) * 1.0
                    avg_cost_m47 = np.average(cost_m47_arr) * 1.0
                    avg_cost_tot = avg_cost_m38 + avg_cost_m47*0
                    if optimizer_state == 0:
                        # In the last state 'nsh_delta' was going up
                        avg_cost_tot_up = avg_cost_tot
                        # Get ready for next iteration
                        optimizer_state = 1
                        nsh_delta = nsh_delta - max_abs_rate
                    elif optimizer_state == 1:
                        # In the last state 'nsh_delta' was going down
                        avg_cost_tot_down = avg_cost_tot
                        # Now it is time to decide where to go
                        avg_cost_tot_diff = avg_cost_tot_up - avg_cost_tot_down
                        if avg_cost_tot_diff < 0.0:
                            nsh_delta = nsh_delta + max_abs_rate
                            print('next I should go up')
                        if avg_cost_tot_diff >= 0.0:
                            nsh_delta = nsh_delta - max_abs_rate
                            print('next I should go down')
                        # Get ready for next iteration
                        optimizer_state = 0
                        nsh_delta = nsh_delta + max_abs_rate
                    else:
                        raise RuntimeError('Bring the vehicle down')

                    # Max rate
                    if (nsh_delta - nsh_delta_prev) > +max_abs_rate:
                        print('max rate was triggered')
                        print('nsh_delta = nsh_delta_prev + max_delta_change')
                        print(f'{nsh_delta}={nsh_delta_prev}+{max_abs_rate}')
                        nsh_delta = nsh_delta_prev + max_abs_rate
                    if (nsh_delta - nsh_delta_prev) < -max_abs_rate:
                        print('max rate was triggered')
                        print('nsh_delta = nsh_delta_prev - max_abs_rate')
                        print(f'{nsh_delta}={nsh_delta_prev}-{max_abs_rate}')
                        nsh_delta = nsh_delta_prev - max_abs_rate
                    # Max abs ranges
                    max_abs_val = 0.9
                    if nsh_delta >= +max_abs_val:
                        nsh_delta = +max_abs_val
                    if nsh_delta <= -max_abs_val:
                        nsh_delta = -max_abs_val

                    print(f'optimizer_state {optimizer_state}, '
                          f'nsh_delta {nsh_delta}')
                    print(f'avg_cost_tot_up {avg_cost_tot_up}, '
                          f'avg_cost_tot_down {avg_cost_tot_down}')

                    nsh_cmd = f'firefly write_delta {nsh_delta} {nsh_delta} 1'
                    fmavl_queue.put(
                        FireflyMavMsg(FireflyMavEnum.nsh_command, nsh_cmd))

                    # Next iteration
                    nsh_delta_prev = nsh_delta
                    avg_cost_tot_prev = avg_cost_tot
                    cost_m38_arr = []
                    cost_m47_arr = []
                    cnt_samples = 0

                optim_data = f'{nsh_delta}, {avg_cost_m38}, {avg_cost_m47}, ' \
                             f'{avg_cost_tot}, {avg_cost_tot_prev}'
                sensor_data = f'{sensor_data}, {fcost}, {optim_data}'
                telem_logger.save_data(log_data=sensor_data, log_header='')

        except KeyboardInterrupt:
            fmavl_queue.put(FireflyMavMsg(FireflyMavEnum.stop_running, True))
            fmavl_thread.join(timeout=60 * 1)


if __name__ == '__main__':
    EscOptimizer.run_optimizer()

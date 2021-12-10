#!/usr/bin/python3
import datetime
import sys
# import numpy as np
# import signal
# import time
# import datetime
# import time

from toopazo_tools.file_folder import FileFolderTools as FFTools
from toopazo_tools.telemetry import TelemetryLogger

from live_ars.ars_interface import ArsIface
from live_esc.kde_uas85uvc.kdecan_interface import KdeCanIface
from ctrlalloc_interface import CtrlAllocIface

# from mavsdk import System
# import asyncio


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


# async def mavsdk_run():
#     drone = System()
#     await drone.connect(system_address="serial:///dev/ttyACM0:115200")
#     print("Waiting for drone ..")
#     async for state in drone.core.connection_state():
#         if state.is_connected:
#             # arg = drone.telemetry.actuator_output_status()
#             arg = drone.info.get_identification()
#             print(f"Drone discovered : {arg}")
#             # print(f"Drone discovered with UUID: {state.uuid}")
#             break


if __name__ == '__main__':
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(mavsdk_run())

    telem_folder = parse_user_arg(sys.argv[1])
    telem_iface = FireflyIfaceWrapper(
        ars_port='/dev/ttyACM0', ctrlalloc_port='/dev/ttyUSB0')
    telem_ext = ".ctrlalloc"
    telem_logger = TelemetryLogger(telem_folder, telem_iface, telem_ext)

    sampling_period = 0.1
    # data = {
    #     'header': {
    #         'sps': 200, 'mills': 9000, 'secs': 9.0, 'dtmills': 9000
    #     },
    #     'data': [0, 0, 0, 0, 0, 0, 0, 0, 510, 520, 507, 548, 10, 14,
    #              17, 19]
    # }
    fields = [
        "sps", "mills", "secs", "dtmills",
        "rpm1", "rpm2", "rpm3", "rpm4",
        "rpm5", "rpm6", "rpm7", "rpm8",
        "cur1", "cur2", "cur3", "cur4",
        "cur5", "cur6", "cur7", "cur8"
    ]
    ars_header = ", ".join(fields)
    esc_header = "time s, escid, "\
                 "voltage V, current A, angVel rpm, temp degC, warning, " \
                 "inthtl us, outthtl perc"
    log_header = f"{ars_header}, {esc_header}"
    telem_logger.live_data(sampling_period, log_header)


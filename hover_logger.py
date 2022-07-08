import os
import sys
import time
import asyncio

import numpy as np

from mavsdk import System

from esc_module import SensorIfaceWrapper, EscOptimizer

vel_limits = ['0.5', '1.0', '1.5', '2', '5', '10']

# set global counter variable to create log saving folder
log_number = 0
stop_flag = False


async def set_stop_flag():
    global stop_flag
    print("Stop flag will be set in 20 seconds!")
    await asyncio.sleep(20)
    stop_flag = True


def create_flight_folder():
    # create the necessary folders for saving .csv log files
    global log_number
    current_path = os.getcwd()
    log_path = current_path + f'/flight_{log_number}'

    # create logs folder
    while os.path.exists(log_path):
        log_number += 1
        log_path = current_path + f'/flight_{log_number}'

    os.mkdir(f'./flight_{log_number}')

    # create sub-folders for all limits
    for limit in vel_limits:
        if not os.path.isdir(log_path + f'/hover_{limit}_limit'):
            os.mkdir(f'{log_path}/hover_{limit}_limit')


class PixhawkConnection:
    esc_interface = None
    pixhawk = System()

    @classmethod
    async def initialize_drone(cls):

        try:
            print("Trying to connect to pixhawk...")
            await asyncio.wait_for(cls.pixhawk.connect(system_address="serial:///dev/ttyUSB0:921600"), timeout=5)

        except asyncio.TimeoutError:
            print("ERROR: Connection to pixhawk failed!")
            return False

        async for state in cls.pixhawk.core.connection_state():
            if state.is_connected:
                print(f"-- Connected to pixhawk!")
                break

        # try to connect to the ESCs
        try:
            cls.esc_interface = SensorIfaceWrapper(ars_port='/dev/ttyACM0')
            cls.esc_data_avail = True
            print("-- ESC data available --> YES")
        except OSError:
            print("-- ESC data available --> NO")
            cls.esc_data_avail = False

        # create folder to save the flight data
        create_flight_folder()

        return True

    @classmethod
    async def log_hovering(cls):
        """ Save 1000 data points to file """
        global stop_flag
        print("Start logging!")

        n_bins = len(vel_limits)

        log_points = 100000
        min_seq_length = 15
        max_seq_length = 300

        data_array = np.zeros((n_bins, max_seq_length, 7+16))
        v_norm_array = np.zeros(100)

        # calculate boundaries for vnorm
        v_i = 14.3452  # calculated induced velocity of vehicle

        limit = np.array([float(a)*v_i*0.01 for a in vel_limits])
        seq_counter = np.zeros(n_bins, dtype=int)
        seq_length = np.zeros(n_bins, dtype=int)

        counter = 0

        start_time = time.time()
        # here starts the loop which reads the data
        async for data in cls.pixhawk.telemetry.odometry():
            begin = time.time()

            if counter % 100 == 0:
                logged_time = time.time()
                v_av = np.sum(v_norm_array)/100
                print(f'Logged {counter} data points in {logged_time - start_time} seconds. Avg. vnorm = {v_av}')
                start_time = logged_time

            # check for interrupt flag
            if stop_flag:
                del cls.pixhawk
                break

            t = data.time_usec
            u = data.velocity_body.x_m_s
            v = data.velocity_body.y_m_s
            w = data.velocity_body.z_m_s

            x = data.position_body.x_m
            y = data.position_body.x_m
            z = data.position_body.x_m

            vnorm = np.sqrt(u ** 2 + v ** 2 + w ** 2)

            v_norm_array[counter%100] = vnorm

            if PixhawkConnection.esc_data_avail:
                sensor_data = PixhawkConnection.sensor_iface.get_data()
                parsed_data = EscOptimizer.parse_sensor_data(sensor_data)

            for i in range(n_bins):
                if vnorm < limit[i] and seq_length[i] < max_seq_length-1:
                    data_array[i, seq_length[i], :7] = np.array([t, x, y, z, u, v, w])

                    if PixhawkConnection.esc_data_avail:
                        for j in range(8):
                            data_array[i, seq_length[i], j+7] = parsed_data[f'voltage_{11+j}']
                            data_array[i, seq_length[i], j+15] = parsed_data[f'current_{11+j}']

                    seq_length[i] += 1

                else:

                    if seq_length[i] >= min_seq_length:
                        seq_start = counter - seq_length[i]
                        seq_end = counter
                        np.savetxt(f'./flight_{log_number}/hover_{vel_limits[i]}_limit/sequence_{seq_counter[i]}_{seq_start}_{seq_end}.csv',
                                   data_array[i, :seq_length[i], :], delimiter=',', comments='',
                                   header='t,x,y,z,u,v,w,U11,U12,U13,U14,U15,U16,U17,U18, \
                                          I11,I12,I13,I14,I15,I16,I17,I18')
                        seq_counter[i] += 1

                    seq_length[i] = 0
                    data_array[i, :, :] = 0

            counter += 1
            end = time.time()


async def control_loop():
    init = await PixhawkConnection.initialize_drone()

    if init:
        hover_task = asyncio.create_task(PixhawkConnection.log_hovering())
        stop_task = asyncio.create_task(set_stop_flag())
        await hover_task
        await stop_task


if __name__ == '__main__':

    print("Data logger has been started!")
    asyncio.run(control_loop())

    print("Finished data logging!")
    sys.exit()

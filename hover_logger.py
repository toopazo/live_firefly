from mavsdk import System
import mavsdk
import asyncio

import os


import numpy as np
import matplotlib.pyplot as plt

import time

limit_text = ['0.5', '1.0', '1.5', '2', '5', '10']

class MavFirefly:

    stop_logging = False

    @staticmethod
    async def initialize_drone():
        drone = System()
        await drone.connect(system_address="serial:///dev/ttyUSB0:921600")

        print("Waiting for drone to connect...")
        async for state in drone.core.connection_state():
            if state.is_connected:
                print(f"-- Connected to drone!")
                break

        async for health in drone.telemetry.health():
            print(f"-- health {health}")
            if health.is_accelerometer_calibration_ok:
                print(f"-- is_accelerometer_calibration_ok {health.is_accelerometer_calibration_ok}")
                break
        try:
            print(f"-- Arming drone")
            data = await drone.action.arm()
            print("data:", data)
        except mavsdk.action.ActionError as error:
            print(f'mavsdk.action.ActionError {error}')

        try:
            info = await drone.info.get_flight_information()
            print("info:", info)
        except mavsdk.info.InfoError as error:
            print(f'mavsdk.info.InfoError {error}')

        # Start the tasks
        print(f"-- Starting tasks")

        try:
            print("Try to set forward speed")
            await drone.action.set_current_speed(5)
        except mavsdk.action.ActionError as error:
            print(f'mavsdk.info.InfoError {error}')

        await MavFirefly.log_hovering(drone.telemetry.odometry)
        loop.stop()

        return drone

    @staticmethod
    async def set_logging():
        MavFirefly.stop_logging = True

    @staticmethod
    async def get_logging():
        return MavFirefly.stop_logging

    @staticmethod
    async def log_hovering(method):
        """ Save 1000 data points to file """

        print("Start logging!")

        n_bins = len(limit_text)

        log_points = 2000
        min_seq_length = 60
        max_seq_length = 300

        data_array = np.zeros((n_bins, max_seq_length, 7))
        v_norm_array = np.zeros(log_points)

        u_array = np.zeros(log_points)
        v_array = np.zeros(log_points)
        w_array = np.zeros(log_points)

        # calculate boundaries for vnorm
        v_i = 14.3452  # calculated induced velocity of vehicle

        #limit = np.array([0.005, 0.01, 0.015]) * v_i
        limit = np.array([float(a) for a in limit_text])
        seq_counter = np.zeros(n_bins, dtype=int)
        seq_length = np.zeros(n_bins, dtype=int)

        counter = 0

        start_time = time.time()
        # here starts the loop which reads the data
        async for data in method():

            if counter % 100 == 0:
                logged_time = time.time()
                print(f'Processed {counter} data points in {logged_time - start_time} seconds')
                start_time = logged_time

            # check for interrupt flag
            if await MavFirefly.get_logging():
                break

            t = data.time_usec
            u = data.velocity_body.x_m_s
            v = data.velocity_body.y_m_s
            w = data.velocity_body.z_m_s

            x = data.position_body.x_m
            y = data.position_body.x_m
            z = data.position_body.x_m

            vnorm = np.sqrt(u ** 2 + v ** 2 + w ** 2)

            v_norm_array[counter] = vnorm
            u_array[counter] = u
            v_array[counter] = v
            w_array[counter] = w

            for i in range(n_bins):
                if vnorm < limit[i] and seq_length[i] < max_seq_length-1:
                    data_array[i, seq_length[i], :] = np.array([t, x, y, z, u, v, w])
                    seq_length[i] += 1

                else:
                    if seq_length[i] >= min_seq_length:
                        seq_start = counter - seq_length[i]
                        seq_end = counter
                        np.savetxt(f'./logs/hover_{limit_text[i]}ms_limit/sequence_{seq_start}_{seq_end}.csv',
                                   data_array[i, :seq_length[0], :], delimiter=',', header='t, x, y, z, u, v, w')
                        seq_counter[i] += 1

                    seq_length[i] = 0
                    data_array[i, :, :] = 0

            counter += 1

            if counter == log_points:
                fig, ax = plt.subplots()
                ax.plot(v_norm_array, label='vnorm')
                print(max(v_norm_array))
                ax.plot(u_array, label='u')
                ax.plot(v_array, label='v')
                ax.plot(w_array, label='w')

                ax.grid()
                ax.legend()
                plt.show()
                break


if __name__ == '__main__':

    asyncio.ensure_future(MavFirefly.initialize_drone())

    # create logging folder
    current_path = os.getcwd()
    log_path = current_path + '/logs'

    # create logs folder
    if not os.path.exists(log_path):
        os.mkdir('logs')

    # create sub-folders for all limits

    for limit in limit_text:
        if not os.path.isdir(log_path + f'/hover_{limit}ms_limit'):
            os.mkdir(f'logs/hover_{limit}ms_limit')

    # Runs the event loop until the program is canceled with e.g. CTRL-C
    loop = asyncio.get_event_loop()
    loop.run_forever()
    print("Stop")
    #end = time.time()
    #print(f'Ended after {end-start} seconds')

    # try to break program with a suitable flag

    # try to access esc motor data
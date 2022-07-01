from mavsdk import System
import mavsdk
import asyncio

import numpy as np
import matplotlib.pyplot as plt

import time


class MavFirefly:

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

        await MavFirefly.log_hovering(drone.telemetry.odometry)
        loop.stop()

        return drone

    @staticmethod
    async def log_hovering(method):
        """ Save 1000 data points to file """

        print("Start logging!")
        log_points = 1000
        min_seq_length = 20
        max_seq_length = 100
        data_array = np.zeros((max_seq_length, 7))
        v_norm_array = np.zeros(log_points)

        u_array = np.zeros(log_points)
        v_array = np.zeros(log_points)
        w_array = np.zeros(log_points)

        limit = 0.07

        no_seq = 0
        seq_length = 0
        counter = 0

        start_time = time.time()
        # here starts the loop which reads the data
        async for data in method():

            if counter % 100 == 0:
                logged_time = time.time()
                print(f'Processed {counter} data points in {logged_time - start_time} seconds')
                start_time = logged_time

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

            if vnorm < limit and seq_length < 50:
                data_array[seq_length, :] = np.array([t, x, y, z, u, v, w])
                seq_length += 1

            else:
                if seq_length >= min_seq_length:

                    np.savetxt(f'./logs/logged_data{no_seq}.csv', data_array[:seq_length, :],
                               delimiter=',', header='t, x, y, z, u, v, w')
                    no_seq += 1

                seq_length = 0
                data_array = np.zeros((100, 7))

            counter += 1

            if counter == log_points:
                fig, ax = plt.subplots()
                ax.plot(v_norm_array, label='vnorm')

                ax.plot(u_array, label='u')
                ax.plot(v_array, label='v')
                ax.plot(w_array, label='w')

                ax.grid()
                ax.legend()
                plt.show()
                break
                

if __name__ == '__main__':

    asyncio.ensure_future(MavFirefly.initialize_drone())

    start = time.time()

    # Runs the event loop until the program is canceled with e.g. CTRL-C
    loop = asyncio.get_event_loop()
    loop.run_forever()
    print("Stop")
    end = time.time()
    print(f'Ended after {end-start} seconds')

import os
import sys
import time
import asyncio

import numpy as np
import pandas as pd

from mavsdk import System

from esc_module import SensorIfaceWrapper, EscOptimizer

from pymavlink import mavutil

vel_limits = ['0.5', '1.0', '1.5', '2', '5', '10']

# set global counter variable to create log saving folder
log_number = 0
stop_flag = False


async def set_stop_flag():
    global stop_flag
    timeout = 20
    print(f"Stop flag will be set in {timeout} seconds!")

    await asyncio.sleep(timeout)
    stop_flag = True
    return 2


async def check_stop():

    global stop_flag
    #counter = 0
    while True:
        armed = await PixhawkConnection.check_armed_state()

        if armed:
            await asyncio.sleep(3)
            continue
        else:
            stop_flag = True
            break


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
    #pixhawk = System()
    pixhawk = mavutil.mavlink_connection(device="/dev/ttyACM0", baud="921600", dialect='development')

    @classmethod
    async def initialize_drone(cls):

        cls.pixhawk.wait_heartbeat()
            #if state.is_connected:
        print(f"-- Connected to pixhawk!")

        msg = cls.pixhawk.recv_match(type='SYS_STATUS', blocking=True)
        if not msg:
            print('Could not get heartbeat message!')
            return 0
        if msg.get_type() == "BAD_DATA":
            if mavutil.all_printable(msg.data):
                sys.stdout.write(msg.data)
                sys.stdout.flush()
        else:
            pass

        # try to connect to the ESCs
        try:
            cls.esc_interface = SensorIfaceWrapper() # port does not matter
            cls.esc_data_avail = True
            print("-- ESC data available --> YES")
        except OSError:
            print("-- ESC data available --> NO")
            cls.esc_data_avail = False

        return 1

    @classmethod
    async def check_armed_state(cls):

        async for armed_state in cls.pixhawk.telemetry.armed():

            if armed_state:
                return 1
            else:
                return 0

    @classmethod
    async def log_hovering(cls):
        """ Save 1000 data points to file """
        global stop_flag
        print("Start logging!")

        n_bins = len(vel_limits)

        log_points = 100000
        min_seq_length = 15
        max_seq_length = 300

        #odometry_data = np.zeros((n_bins, max_seq_length, 7))
        #motor_data = np.zeros((n_bins, max_seq_length, 40))
        # data_array = np.zeros((n_bins, max_seq_length, 47)
        data_array = np.zeros((n_bins, max_seq_length, 47+35))
        v_norm_array = np.zeros(100)

        # calculate boundaries for vnorm
        v_i = 14.3452  # calculated induced velocity of vehicle

        limit = np.array([float(a)*v_i*0.01 for a in vel_limits])
        seq_counter = np.zeros(n_bins, dtype=int)
        seq_length = np.zeros(n_bins, dtype=int)

        counter = 0

        start_time = time.time()
        # here starts the loop which reads the data
        #async for data in cls.pixhawk.telemetry.odometry():

        while True:

            odometry = cls.pixhawk.recv_match(type='ODOMETRY', blocking=True)
            # timesync = cls.pixhawk.recv_match(type='TIMESYNC', blocking=True)

            """ Faked controll allocation"""
            # ctrl_alloc = cls.pixhawk.recv_match(type='FIREFLY_CTRLALLOC', blocking=True)
            ctrl_alloc = {'time_boot_ms': 99770, 'status': 1, 'nooutputs': 8,
                          'controls': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                          'output': [-1, -1, -1, -1, -1, -1, -1, -1],
                          'pwm_limited': [900, 901, 902, 903, 904, 905, 906, 907, 908],
                          'delta': [0, 1, 2, 3, 4, 5, 6, 7]}

            # Control message for logging process
            if counter % 100 == 0:
                logged_time = time.time()
                v_av = np.sum(v_norm_array)/100
                print(f'Logged {counter} data points after {logged_time - start_time} seconds. Avg. vnorm = {v_av}')
                start_time = logged_time

            t = odometry.time_usec
            u = odometry.vx
            v = odometry.vy
            w = odometry.vz

            x = odometry.x
            y = odometry.y
            z = odometry.z

            # variables for control allocation
            t_ctrl = ctrl_alloc['time_boot_ms']
            ctrl_status = ctrl_alloc['status']
            n_out = ctrl_alloc['nooutputs']
            controls = ctrl_alloc['controls']
            ctrl_output = ctrl_alloc['output']
            pwm_limit = ctrl_alloc['pwm_limited']
            delta = ctrl_alloc['delta']



            vnorm = np.sqrt(u ** 2 + v ** 2 + w ** 2)

            v_norm_array[counter % 100] = vnorm

            if PixhawkConnection.esc_data_avail:
                try:
                    sensor_data = PixhawkConnection.esc_interface.get_data()
                    parsed_data = EscOptimizer.parse_sensor_data(sensor_data)

                    if counter % 100 == 0:
                        print(parsed_data)

                except TypeError:
                    print("Could not retrieve ESC data! Check if ESCs are powered!")
                    PixhawkConnection.esc_data_avail = False

            for i in range(n_bins):
                if vnorm < limit[i] and seq_length[i] < max_seq_length-1 and not stop_flag:
                    data_array[i, seq_length[i], :7] = np.array([t, x, y, z, u, v, w])

                    # fake data
                    data_array[i, seq_length[i], 47] = t_ctrl
                    data_array[i, seq_length[i], 48] = ctrl_status
                    data_array[i, seq_length[i], 49] = n_out

                    for j in range(7, 15):
                        if PixhawkConnection.esc_data_avail:
                            data_array[i, seq_length[i], j] = parsed_data[f'voltage_{4+j}']  # 7-14
                            data_array[i, seq_length[i], j + 8] = parsed_data[f'current_{4+j}']  # 15-22
                            data_array[i, seq_length[i], j + 16] = parsed_data[f'angVel_{4+j}']  # 23-30
                            data_array[i, seq_length[i], j + 24] = parsed_data[f'inthtl_{4 + j}']  # 31-38
                            data_array[i, seq_length[i], j + 32] = parsed_data[f'outthtl_{4 + j}'] # 39-46
                        data_array[i, seq_length[i], j + 43] = controls[j - 7]  # 49-56
                        data_array[i, seq_length[i], j + 51] = ctrl_output[j - 7]  # 58-65
                        data_array[i, seq_length[i], j + 59] = pwm_limit[j - 7]  # 66-73
                        data_array[i, seq_length[i], j + 67] = delta[j - 7]  # 74-81

                    seq_length[i] += 1

                else:

                    if seq_length[i] >= min_seq_length:
                        seq_start = counter - seq_length[i]
                        seq_end = counter
                        #dataframe = pd.DataFrame(motor_data, columns=['x,y,z,u,v,w'])
                        np.savetxt(f'./flight_{log_number}/hover_{vel_limits[i]}_limit/sequence_{seq_counter[i]}_{seq_start}_{seq_end}.csv',
                                   data_array[i, :seq_length[i], :], delimiter=',', comments='',
                                   header='t,x,y,z,u,v,w,'
                                          'U11,U12,U13,U14,U15,U16,U17,U18,'
                                          'I11,I12,I13,I14,I15,I16,I17,I18,'
                                          'omega1,omega2,omega3,omega4,omega5,omega6,omega7,omega8,'
                                          'thr_in1,thr_in2,thr_in3,thr_in4,thr_in5,thr_in6,thr_in7,thr_in8,'
                                          'thr_out1,thr_out2,thr_out3,thr_out4,thr_out5,thr_out6,thr_out7,thr_out8,'
                                          'time_boot_ms,status,nooutputs,'
                                          'ctrl_1, ctrl_2, ctrl_3, ctrl_4, ctrl_5, ctrl_6, ctrl_7, ctrl_8,'
                                          'output1, output2, output3, output4, output5, output6, output7, output8,'
                                          'pwm_1, pwm_2, pwm_3, pwm_4, pwm_5, pwm_6, pwm_7, pwm_8,'
                                          'delta_1, delta_2, delta_3, delta_4, delta_5, delta_6, delta_7, delta_8')

                        seq_counter[i] += 1

                    seq_length[i] = 0
                    data_array[i, :, :] = 0

            # check for interrupt flag
            if stop_flag:
                # del cls.pixhawk
                stop_flag = False
                return 1
                break

            counter += 1


async def control_loop():
    # implements finite state machine to control data logging process, depending on whether vehicle is armed or not

    state = 0  # init state

    while True:

        if state == 0:
            print("State 0")
            # try to connect to Pixhawk
            connected = await PixhawkConnection.initialize_drone()

            if connected:
                state = 2
            else:
                await asyncio.sleep(5)

        if state == 1:
            print("State 1")
            # check if vehicle is armed
            #armed = await PixhawkConnection.check_armed_state()
            armed = 1 # comment out to use real armed flag
            #if armed:
            #    print(f"-- Vehicle armed -> start logging")
            #    state = 2
            #else:
            #    print("Vehicle not armed -> waiting for armed state")
            #    await asyncio.sleep(5)

        if state == 2:
            print("State 2")

            create_flight_folder() # create folder to save the flight data
            hover_task = asyncio.create_task(PixhawkConnection.log_hovering())
            #stop_task = asyncio.create_task(check_stop()) # comment in for real flight
            #stop_task = asyncio.create_task(set_stop_flag()) # comment in for real flight
            state = 3

        if state == 3:
            print("State 3")
            state = await hover_task

if __name__ == '__main__':

    print("Data logger has been started!")
    mavutil.set_dialect('development')
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(control_loop())
    except KeyboardInterrupt:
        print("Stopped Program with Keyboard Interrupt")
        del PixhawkConnection.pixhawk
        loop.stop()

    print("Finished data logging!")
    sys.exit()

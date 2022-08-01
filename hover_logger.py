import os
import sys
import time

import asyncio
from aioconsole import ainput

import numpy as np

from mavsdk import System

from esc_module import SensorIfaceWrapper, EscOptimizer
from firefly_mavlink_live import FireflyMavlink, MavlinkSerialPort

from timeit import default_timer as timer

from pymavlink import mavutil
import serial

vel_limits = ['0.5', '1.0', '1.5', '2', '5', '10']

# set global counter variable to create log saving folder
log_number = 0
stop_flag = False
supress_output = False  # set that flag if nsh shell is open


async def check_stop():

    global stop_flag

    while True:
        armed = await PixhawkConnection.check_armed_state()
        PixhawkConnection.armed = armed
        if armed:
            await asyncio.sleep(3)
            continue
        else:
            await asyncio.sleep(5)  # wait 5 second to make sure, that last part of flight is logged
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
    pixhawk = System()
    nsh_cmd = f'firefly write_delta 0 0 1'
    logger_cmd = '+0.0 +0.0'
    armed = 0
    @classmethod
    async def initialize_drone(cls):

        try:
            print("Trying to connect to pixhawk...")
            await asyncio.wait_for(cls.pixhawk.connect(system_address="serial:///dev/ttyUSB0:921600"), timeout=3)

        except asyncio.TimeoutError:
            print("ERROR: Connection to pixhawk failed!")
            return 0

        async for state in cls.pixhawk.core.connection_state():

            if state.is_connected:
                print(f"-- Connected to pixhawk!")
                break
            else:
                continue
                #return 0

        # try to connect to the ESCs
        try:
            cls.esc_interface = SensorIfaceWrapper() # port does not matter
            cls.esc_data_avail = True
            print("-- ESC data available --> YES")
        except OSError:
            print("-- ESC data available --> NO")
            cls.esc_data_avail = False


        fm = FireflyMavlink(port='/dev/ttyACM0', baudrate=57600)
        cls.mav_serial = MavlinkSerialPort(fm.port, fm.baudrate, devnum=10)

        return 1

    @classmethod
    async def check_armed_state(cls):

        async for armed_state in cls.pixhawk.telemetry.armed():

            if armed_state:
                return 1
            else:
                return 0

    @classmethod
    async def set_nsh_command(cls):
        """ Get nsh command and send it to pixhawk """

        print("Get input!")
        global supress_output
        global stop_flag

        while cls.armed:
            old_cmd = cls.nsh_cmd
            supress_output = True
            line = await ainput(">>> ")
            print(f'Received the following input: {line}')


            #TODO: check if nsh command has right format

            try:

                #if len(line) != 3:
                #    raise ValueError

                #cmd_1 = float(line[0:3])
                #cmd_2 = float(line[4:])

                cls.nsh_cmd = f'firefly write_delta {line} 1'
                print(f'sent cmd: {cls.nsh_cmd}')
                cls.logger_cmd = line
            except ValueError:
                print("Faulty input! Try again!")
                continue
            #cls.mav_serial.write('\n')

            try:
                next_heartbeat_time = timer()

                #while True:
                #try:
                    # fm_msg = _queue.get(block=False)
                    #fm_msg = _queue.get(block=True, timeout=0.01)
                    #assert isinstance(fm_msg, FireflyMavMsg)
                    # print(f'A {FireflyMavMsg.__name__} was received')
                    #if fm_msg.key == FireflyMavEnum.stop_running and fm_msg.val:
                        #mav_serial.close()
                        #return
                    #if fm_msg.key == FireflyMavEnum.nsh_command:
                        #cmd = fm_msg.val
                if not old_cmd == cls.nsh_cmd:
                    cls.mav_serial.write(cls.nsh_cmd + '\n')
                #print(cls.nsh_cmd)

                data = cls.mav_serial.read(4096 * 12)
                while data != "": # Flush buffer
                    #print(data)
                    data = cls.mav_serial.read(4096 * 12)
                #print(data)
                #if data and len(data) > 0:
                #    print(data, end='')

                # handle heartbeat sending
                heartbeat_time = timer()
                if heartbeat_time > next_heartbeat_time:
                    cls.mav_serial.mav.mav.heartbeat_send(
                        mavutil.mavlink.MAV_TYPE_GCS,
                        mavutil.mavlink.MAV_AUTOPILOT_GENERIC, 0, 0, 0
                    )
                    next_heartbeat_time = heartbeat_time + 1
                supress_output = False
                await asyncio.sleep(10)
            except serial.serialutil.SerialException as e:
                print(e)

        return 1

    @classmethod
    async def log_hovering(cls):
        """ Save 1000 data points to file """
        global stop_flag
        global supress_output
        print("Start logging!")

        n_bins = len(vel_limits)

        min_seq_length = 15
        max_seq_length = 500

        data_array = np.zeros((n_bins, max_seq_length, 47 + 3 + 35 + 2))
        v_norm_array = np.zeros(100)

        # calculate boundaries for vnorm
        v_i = 14.3452  # calculated induced velocity of vehicle

        limit = np.array([float(a)*v_i*0.01 for a in vel_limits])
        seq_counter = np.zeros(n_bins, dtype=int)
        seq_length = np.zeros(n_bins, dtype=int)

        counter = 0

        start_time = time.time()
        control_topic = cls.pixhawk.telemetry.actuator_control_target()
        control_data = await control_topic.__anext__()

        async for data in cls.pixhawk.telemetry.odometry():

            if counter % 3 == 0 and counter != 0:
                control_data = await control_topic.__anext__()

            """ Faked control allocation data -> needs to be replaced with real data from topic"""
            # ctrl_alloc = cls.pixhawk.recv_match(type='FIREFLY_CTRLALLOC', blocking=True)
            ctrl_alloc = {'time_boot_ms': 99770, 'status': 1, 'nooutputs': 8,
                          'controls': control_data.controls,
                          #'controls': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                          'output': [-1, -1, -1, -1, -1, -1, -1, -1],
                          'pwm_limited': [900, 901, 902, 903, 904, 905, 906, 907, 908],
                          'delta': [0, 1, 2, 3, 4, 5, 6, 20]}

            # Control message for logging process
            if counter % 100 == 0:
                logged_time = time.time()
                v_av = np.sum(v_norm_array)/100

                #if not supress_output:
                    #print(f'Logged {counter} data points after {logged_time - start_time} seconds. Avg. vnorm = {v_av} - Current nsh-cmd: {PixhawkConnection.nsh_cmd}')
                start_time = logged_time

            t = data.time_usec

            u = data.velocity_body.x_m_s
            v = data.velocity_body.y_m_s
            w = data.velocity_body.z_m_s

            x = data.position_body.x_m
            y = data.position_body.y_m
            z = data.position_body.z_m

            p = data.angular_velocity_body.roll_rad_s
            q = data.angular_velocity_body.pitch_rad_s
            r = data.angular_velocity_body.yaw_rad_s

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

                except TypeError:
                    print("Could not retrieve ESC data! Check if ESCs are powered!")
                    PixhawkConnection.esc_data_avail = False

            for i in range(n_bins):
                if vnorm < limit[i] and seq_length[i] < max_seq_length-1 and not stop_flag:
                    data_array[i, seq_length[i], :10] = np.array([t, x, y, z, u, v, w, p, q, r])  # 0-9

                    # fake data
                    data_array[i, seq_length[i], 50] = t_ctrl
                    data_array[i, seq_length[i], 51] = ctrl_status
                    data_array[i, seq_length[i], 52] = n_out

                    data_array[i, seq_length[i], -2] = float(cls.logger_cmd[0:4])
                    data_array[i, seq_length[i], -1] = float(cls.logger_cmd[5:9])

                    for j in range(10, 18):
                        if PixhawkConnection.esc_data_avail:
                            data_array[i, seq_length[i], j] = parsed_data[f'voltage_{1 + j}']  # 10-17
                            data_array[i, seq_length[i], j + 8] = parsed_data[f'current_{1 + j}']  # 18-25
                            data_array[i, seq_length[i], j + 16] = parsed_data[f'angVel_{1 + j}']  # 26-33
                            data_array[i, seq_length[i], j + 24] = parsed_data[f'inthtl_{1 + j}']  # 34-41
                            data_array[i, seq_length[i], j + 32] = parsed_data[f'outthtl_{1 + j}']  # 42-49

                        data_array[i, seq_length[i], j + 43] = controls[j - 10]  # 53-60
                        data_array[i, seq_length[i], j + 51] = ctrl_output[j - 10]  # 61-68
                        data_array[i, seq_length[i], j + 59] = pwm_limit[j - 10]  # 69-76
                        data_array[i, seq_length[i], j + 67] = delta[j - 10]  # 77-84

                    seq_length[i] += 1

                else:

                    if seq_length[i] >= min_seq_length:

                        seq_start = counter - seq_length[i]
                        seq_end = counter

                        np.savetxt(f'./flight_{log_number}/hover_{vel_limits[i]}_limit/sequence_{seq_counter[i]}_{seq_start}_{seq_end}.csv',
                                   data_array[i, :seq_length[i], :], delimiter=',', comments='',
                                   header='t,x,y,z,u,v,w,p,q,r,'
                                          'U11,U12,U13,U14,U15,U16,U17,U18,'
                                          'I11,I12,I13,I14,I15,I16,I17,I18,'
                                          'omega1,omega2,omega3,omega4,omega5,omega6,omega7,omega8,'
                                          'thr_in1,thr_in2,thr_in3,thr_in4,thr_in5,thr_in6,thr_in7,thr_in8,'
                                          'thr_out1,thr_out2,thr_out3,thr_out4,thr_out5,thr_out6,thr_out7,thr_out8,'
                                          'time_boot_ms,status,nooutputs,'
                                          'ctrl_1,ctrl_2,ctrl_3,ctrl_4,ctrl_5,ctrl_6,ctrl_7,ctrl_8,'
                                          'output1,output2,output3,output4,output5,output6,output7,output8,'
                                          'pwm_1,pwm_2,pwm_3,pwm_4,pwm_5,pwm_6,pwm_7,pwm_8,'
                                          'delta_1,delta_2,delta_3,delta_4,delta_5,delta_6,delta_7,delta_8,'
                                          'nsh[0], nsh[1]')

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
            #connected = 1
            if connected:
                state = 1
            else:
                await asyncio.sleep(5)

        if state == 1:
            print("State 1")
            # check if vehicle is armed
            armed = await PixhawkConnection.check_armed_state()
            #armed = 1 # comment out to use real armed flag
            PixhawkConnection.armed = armed
            if armed:
                print(f"-- Vehicle armed -> start logging")
                state = 2

            else:
                print("Vehicle not armed -> waiting for armed state")
                await asyncio.sleep(5)

        if state == 2:
            print("State 2")

            create_flight_folder() # create folder to save the flight data
            hover_task = asyncio.create_task(PixhawkConnection.log_hovering())
            stop_task = asyncio.create_task(check_stop())  # comment in for real flight
            nsh_cmd_task = asyncio.create_task(PixhawkConnection.set_nsh_command())
            state = 3

        if state == 3:
            print("State 3")
            state = await hover_task
            #nsh_return = nsh_cmd_task

if __name__ == '__main__':

    print("Data logger has been started!")
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(control_loop())
    except KeyboardInterrupt:
        print("Stopped Program with Keyboard Interrupt")
        del PixhawkConnection.pixhawk
        PixhawkConnection.mav_serial.close()
        loop.stop()

    print("Finished data logging!")
    sys.exit()

import os
import sys
import time

import asyncio
from aioconsole import ainput

import numpy as np

from mavsdk import System

from esc_connection import SensorIfaceWrapper
from mavlink_serial_connection import MavlinkSerialPort, check_nsh_input

from timeit import default_timer as timer

from pymavlink import mavutil
import serial

vel_limits = ['0.5', '1.0', '1.5', '2', '5', '10']

# set global counter variable to create log saving folder
log_number = 0
supress_output = False  # set that flag if nsh shell is open


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

    # create logfile for console output
    with open(f'{log_path}/console_output.txt', 'w') as fp:
        pass


def print_to_logfile(msg):

    current_path = os.getcwd()
    log_path = current_path + f'/flight_{log_number}'

    with open(f'{log_path}/console_output.txt', 'a') as fp:
        fp.write(msg)


class DataLogger:
    esc_interface = None
    stop_flag = False
    telem2 = System()
    mav_serial = None
    current_delta0 = [0.0, 0.0]
    armed = 0
    reset_trigger = False

    @classmethod
    async def connect_to_pixhawk(cls):

        # connect to TELEM2 port
        try:
            print("Connecting to TELEM2 port -> ", end=" ")
            await asyncio.wait_for(cls.telem2.connect(system_address="serial:///dev/ttyUSB0:921600"), timeout=3)

        except asyncio.TimeoutError:
            print("FAILURE (Timeout)", end="\r")
            return 0

        async for state in cls.telem2.core.connection_state():

            if state.is_connected:
                print("SUCCESS")
                break
            else:
                print("FAILURE (Faulty connection state)")
                return 0

        # connect to the ESCs
        try:
            print("Connecting to ESCs -> ", end=" ")
            cls.esc_interface = SensorIfaceWrapper()  # port does not matter
            cls.esc_data_avail = True
            print("SUCCESS")

        except OSError:
            print("FAILURE")
            cls.esc_data_avail = False

        # connect to the UDP port
        try:
            print("Connecting to UDP port -> ", end=" ")
            #fm = FireflyMavlink(port='/dev/ttyACM0', baudrate=57600)
            cls.mav_serial = MavlinkSerialPort(portname='/dev/ttyACM0', baudrate=57600, devnum=10)
            print("SUCCESS")

        except serial.serialutil.SerialException:
            print("FAILURE (Invalid port)")
            cls.mav_serial = None

        return 1

    @classmethod
    async def check_armed_state(cls, debug=False):

        async for armed_state in cls.telem2.telemetry.armed():

            if debug:
                armed_state = True

            if armed_state:

                cls.armed = 1
                return 1
            else:
                print("WAITING for vehicle to be armed", end="\r")
                cls.armed = 0
                await asyncio.sleep(5)
                return 0

    @classmethod
    async def check_stop(cls):

        while True:
            armed = await cls.check_armed_state()
            cls.armed = armed

            if armed:
                await asyncio.sleep(1)
                continue
            else:
                await asyncio.sleep(5)  # wait 5 second to make sure, that last part of flight is logged
                cls.stop_flag = True
                break

    @classmethod
    async def poll_nsh_cmd(cls):
        """ Get nsh command and send it to telem2 """

        global supress_output
        reset_counter = 0

        while cls.armed:
            supress_output = True

            # await user input
            cmd_input = await ainput(">>> ")
            #cmd_input = "0 0"

            # check for command to continue
            if cmd_input == "":
                supress_output = False
                reset_counter += 1

                if reset_counter >= 3:
                    cls.reset_trigger = True
                    print("Reset triggered!")
                    await cls.reset_delta()
                else:
                    await asyncio.sleep(0.5)

                continue
            reset_counter = 0

            # check that nsh command has right format
            parsed_cmd, valid, message = check_nsh_input(cmd_input, cls.current_delta0)

            # check for command to sweep through delta0 space
            try:
                sweep = cmd_input.split()

                if sweep[0] == "sweep":

                    timestep = float(sweep[1])

                    if timestep < 0 or np.abs(timestep) < 0.2:
                        print("Timestep must be positive and larger than 0.2 seconds!")
                        continue

                    if cls.current_delta0 != [0.0, 0.0]:
                        print("Current delta0 not equal to 0! Reset first!")
                        continue

                    else:
                        print("Start sweep... Press 3x Enter to interrupt!")
                        asyncio.ensure_future(cls.make_sweep(timestep))
                        continue

            except IndexError:
                message = "Missing steptime for sweep! Use 'sweep [TIME BETWEEN STEPS]'"

            if valid:
                cls.send_nsh_cmd(parsed_cmd)
                supress_output = False
                await asyncio.sleep(1)

            else:
                print(message)
                supress_output = False
                await asyncio.sleep(1)
        return 1

    @classmethod
    async def make_sweep(cls, timestep):
        cmd = 0.0
        # sweep down
        while cmd > -1 and not cls.reset_trigger:
            cmd = np.around((cmd - 0.1), decimals=2)
            cls.send_nsh_cmd([cmd, cmd], True)
            await asyncio.sleep(timestep)
        cmd = -1.0

        # sweep up
        while cmd < 0 and not cls.reset_trigger:
            cmd = np.around((cmd + 0.1), decimals=2)
            cls.send_nsh_cmd([cmd, cmd], True)
            await asyncio.sleep(timestep)
        print("Sweep complete!")

    @classmethod
    async def reset_delta(cls):
        cmd = [0, 0]

        reset_cmd_0 = True
        reset_cmd_1 = True

        while reset_cmd_0 and reset_cmd_1:

            reset_cmd_0 = np.around(np.abs(cls.current_delta0[0]), decimals=1) > 0
            reset_cmd_1 = np.around(np.abs(cls.current_delta0[1]), decimals=1) > 0

            if reset_cmd_0:
                cmd[0] = cls.current_delta0[0] - np.sign(cls.current_delta0[0]) * 0.1

            if reset_cmd_1:
                cmd[1] = cls.current_delta0[1] - np.sign(cls.current_delta0[1]) * 0.1

            cls.send_nsh_cmd([cmd[0], cmd[1]])
            await asyncio.sleep(0.3)

        # send [0, 0] as final command to make sure delta0 is reset to [0, 0]
        cls.send_nsh_cmd([0, 0])
        cls.reset_trigger = False

    @classmethod
    def send_nsh_cmd(cls, cmd, verbose=True):

        nsh_cmd = f'firefly write_delta {cmd[0]} {cmd[1]} 1'

        try:
            next_heartbeat_time = timer()

            # check if
            if not cmd == cls.current_delta0:
                cls.mav_serial.write(nsh_cmd + '\n')

                if verbose:
                    print(f'sent cmd: {nsh_cmd}')

            data = cls.mav_serial.read(4096 * 12)
            while data != "":  # Flush buffer
                data = cls.mav_serial.read(4096 * 12)

            # handle heartbeat sending
            heartbeat_time = timer()
            if heartbeat_time > next_heartbeat_time:
                cls.mav_serial.mav.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_GCS,
                    mavutil.mavlink.MAV_AUTOPILOT_GENERIC, 0, 0, 0
                )
                next_heartbeat_time = heartbeat_time + 1

            cls.current_delta0 = cmd
        except serial.serialutil.SerialException as e:
            print(e)

    @classmethod
    async def log_hovering(cls):
        """ Save 1000 data points to file """
        # global stop_flag
        global supress_output
        # print("Start logging!")

        n_bins = len(vel_limits)

        min_seq_length = 15
        max_seq_length = 500

        data_array = np.zeros((n_bins, max_seq_length, 47 + 3 + 35 + 2))
        v_norm_array = np.zeros(100)

        # calculate boundaries for vnorm
        v_i = 14.3452  # calculated induced velocity of vehicle

        limit = np.array([float(a) * v_i * 0.01 for a in vel_limits])
        seq_counter = np.zeros(n_bins, dtype=int)
        seq_length = np.zeros(n_bins, dtype=int)

        counter = 0

        start_time = time.time()
        previous_time = start_time
        control_topic = cls.telem2.telemetry.actuator_control_target()
        control_data = await control_topic.__anext__()

        async for data in cls.telem2.telemetry.odometry():

            if counter % 3 == 0 and counter != 0:
                control_data = await control_topic.__anext__()

            """ Faked control allocation data -> needs to be replaced with real data from topic"""
            # ctrl_alloc = cls.telem2.recv_match(type='FIREFLY_CTRLALLOC', blocking=True)
            ctrl_alloc = {'time_boot_ms': 99770, 'status': 1, 'nooutputs': 8,
                          'controls': control_data.controls,
                          # 'controls': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                          'output': [-1, -1, -1, -1, -1, -1, -1, -1],
                          'pwm_limited': [900, 901, 902, 903, 904, 905, 906, 907, 908],
                          'delta': [0, 1, 2, 3, 4, 5, 6, 20]}

            # Control message for logging process
            if counter % 100 == 0:
                logged_time = time.time()
                v_av = np.sum(v_norm_array) / 100

                #if not supress_output:
                msg = f'{logged_time - start_time}s elapsed! Logged {counter} data points. Took {logged_time - previous_time} seconds for the last 100 data points\n'
                print_to_logfile(msg)
                    #print(f'Logged {counter} data points after {logged_time - start_time} seconds. Avg. vnorm = {v_av}')
                previous_time = logged_time

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

            if DataLogger.esc_data_avail:
                try:
                    sensor_data = cls.esc_interface.get_data()
                    parsed_data = cls.esc_interface.parse_sensor_data(sensor_data)

                except TypeError:
                    print("Could not retrieve ESC data! Check if ESCs are powered!")
                    DataLogger.esc_data_avail = False

            for i in range(n_bins):
                if vnorm < limit[i] and seq_length[i] < max_seq_length - 1 and not cls.stop_flag:
                    data_array[i, seq_length[i], :10] = np.array([t, x, y, z, u, v, w, p, q, r])  # 0-9

                    # fake data
                    data_array[i, seq_length[i], 50] = t_ctrl
                    data_array[i, seq_length[i], 51] = ctrl_status
                    data_array[i, seq_length[i], 52] = n_out

                    data_array[i, seq_length[i], -2] = float(cls.current_delta0[0])
                    data_array[i, seq_length[i], -1] = float(cls.current_delta0[1])

                    for j in range(10, 18):
                        if DataLogger.esc_data_avail:
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

                        np.savetxt(
                            f'./flight_{log_number}/hover_{vel_limits[i]}_limit/sequence_{seq_counter[i]}_{seq_start}_{seq_end}.csv',
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
            if cls.stop_flag:
                cls.stop_flag = False
                return 1

            counter += 1


async def control_loop():
    # implements finite state machine to control data logging process, depending on whether vehicle is armed or not

    state = 0  # init state

    while True:

        if state == 0:

            # try to connect to Pixhawk
            connected = await DataLogger.connect_to_pixhawk()
            # connected = 1

            if connected:
                state = 1
            else:
                await asyncio.sleep(5)

        if state == 1:
            print("Check for arming status -> ", end=" ")
            # check if vehicle is armed
            state = await DataLogger.check_armed_state(debug=False) + 1

        if state == 2:
            print(f"ARMED")
            print("Create flight folder and start logging...")

            create_flight_folder()  # create folder to save the flight data

            # create tasks for logging, stopping and sending nsh command
            hover_task = asyncio.create_task(DataLogger.log_hovering())
            stop_task = asyncio.create_task(DataLogger.check_stop())  # comment in for real flight
            nsh_cmd_task = asyncio.create_task(DataLogger.poll_nsh_cmd())
            state = 3

        if state == 3:
            print("Logger running...")
            try:
                #print("Inside try!")
                #L = await asyncio.gather(DataLogger.log_hovering(), DataLogger.poll_nsh_cmd())
                nsh_result = await nsh_cmd_task
                state = await hover_task
                stop = await stop_task

                #stop = await stop_task
                print("Finished try!")
            except KeyboardInterrupt:
                print("Catched!")


if __name__ == '__main__':

    print("Data logger has been started!")
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(control_loop())
    except KeyboardInterrupt:
        print("")
        print("Stopped Program with Keyboard Interrupt")
        del DataLogger.telem2

        if hasattr(DataLogger, 'mavserial'):
            DataLogger.mav_serial.close()

        loop.stop()

    print("Finished data logging!")
    sys.exit()

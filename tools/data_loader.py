""" Some helper functions to get data from firefly_df """

import numpy as np
import pandas as pd

import os
import glob

from tools.helper_functions import moving_average


def get_current(firefly_df):

    current = {'11': firefly_df['esc11_current'].values, '16': firefly_df['esc16_current'].values,
               '12': firefly_df['esc12_current'].values, '15': firefly_df['esc15_current'].values,
               '13': firefly_df['esc13_current'].values, '18': firefly_df['esc18_current'].values,
               '14': firefly_df['esc14_current'].values, '17': firefly_df['esc17_current'].values}

    return current


def get_voltage(firefly_df):

    voltage = {'11': firefly_df['esc11_voltage'].values, '16': firefly_df['esc16_voltage'].values,
               '12': firefly_df['esc12_voltage'].values, '15': firefly_df['esc15_voltage'].values,
               '13': firefly_df['esc13_voltage'].values, '18': firefly_df['esc18_voltage'].values,
               '14': firefly_df['esc14_voltage'].values, '17': firefly_df['esc17_voltage'].values}

    return voltage


def get_rpm(firefly_df):

    rpm = {'11': firefly_df['esc11_rpm'].values, '16': firefly_df['esc16_rpm'].values,
           '12': firefly_df['esc12_rpm'].values, '15': firefly_df['esc15_rpm'].values,
           '13': firefly_df['esc13_rpm'].values, '18': firefly_df['esc18_rpm'].values,
           '14': firefly_df['esc14_rpm'].values, '17': firefly_df['esc17_rpm'].values}
    return rpm


def get_delta_rpm(firefly_df, smooth=5):

    rpm = get_rpm(firefly_df)

    delta_rpm = {'16': moving_average(rpm['11'], smooth) - moving_average(rpm['16'], smooth),
                 '25': moving_average(rpm['12'], smooth) - moving_average(rpm['15'], smooth),
                 '38': moving_average(rpm['13'], smooth) - moving_average(rpm['18'], smooth),
                 '47': moving_average(rpm['14'], smooth) - moving_average(rpm['17'], smooth),
                 }
    return delta_rpm


def get_power(firefly_df):

    current = get_current(firefly_df)
    voltage = get_voltage(firefly_df)

    individual_power = {'11': voltage['11'] * current['11'], '16': voltage['16'] * current['16'],
                   '12': voltage['12'] * current['12'], '15': voltage['15'] * current['15'],
                   '13': voltage['13'] * current['13'], '18': voltage['18'] * current['18'],
                   '14': voltage['14'] * current['14'], '17': voltage['17'] * current['17']}

    pair_power = {'16': (individual_power['11'] + individual_power['16']),
                  '25': (individual_power['12'] + individual_power['15']),
                  '38': (individual_power['13'] + individual_power['18']),
                  '47': (individual_power['14'] + individual_power['17'])}

    upper_power = {'11': individual_power['11'], '12': individual_power['12'],
                   '13': individual_power['13'], '14': individual_power['14']}

    lower_power = {'15': individual_power['15'], '16': individual_power['16'],
                   '17': individual_power['17'], '18': individual_power['18']}

    return individual_power, pair_power, upper_power, lower_power


def get_in_throttle(firefly_df):

    in_throttle = {'11': firefly_df['esc11_inthrottle'].values, '16': firefly_df['esc16_inthrottle'].values,
                   '12': firefly_df['esc12_inthrottle'].values, '15': firefly_df['esc15_inthrottle'].values,
                   '13': firefly_df['esc13_inthrottle'].values, '18': firefly_df['esc18_inthrottle'].values,
                   '14': firefly_df['esc14_inthrottle'].values, '17': firefly_df['esc17_inthrottle'].values}

    return in_throttle


def get_out_throttle(firefly_df):
    out_throttle = {'11': firefly_df['esc11_outthrottle'].values, '16': firefly_df['esc16_outthrottle'].values,
                    '12': firefly_df['esc12_outthrottle'].values, '15': firefly_df['esc15_outthrottle'].values,
                    '13': firefly_df['esc13_outthrottle'].values, '18': firefly_df['esc18_outthrottle'].values,
                    '14': firefly_df['esc14_outthrottle'].values, '17': firefly_df['esc17_outthrottle'].values}

    return out_throttle


# def moving_average(in_dict, window=5):
#     """ Calculates the moving average of the values for every dictionary"""
#     out_dict = {}
#
#     # do moving average for numpy array
#     if type(in_dict) is np.ndarray:
#         return np.convolve(in_dict, np.ones(window), 'same') / window
#
#     for motor in in_dict:
#         out_dict[motor] = np.convolve(in_dict[motor], np.ones(window), 'same') / window
#
#     return out_dict

def load_flight_data(flight, limit=10):

    data_directory = flight + f'hover_{limit}_limit'
    csv_files = glob.glob(os.path.join(data_directory, "*.csv"))

    if limit == 10 and not csv_files:
        data_directory = flight + f'hover_{100}_limit'
        csv_files = glob.glob(os.path.join(data_directory, "*.csv"))

    sorted_files = []

    # sort sequences in ascending order
    for i in range(len(csv_files)):
        for e in csv_files:
            if f'sequence_{i}_' in e:
                # print(e)
                sorted_files.append(e)
                break

    full_flight = pd.DataFrame()

    for csv_file in sorted_files:

        full_flight = pd.concat([full_flight, pd.read_csv(csv_file)])

    return full_flight


def select_sequence(data, xMin, xMax):

    lower_index = xMin * 30
    upper_index = xMax * 30

    hover = np.arange(len(data))[lower_index: -upper_index]

    return data.iloc[hover]


def clean_data(flightdata):
    # cleaning and renaming of flight_data dataframe

    flightdata = flightdata.rename(columns={'nsh[0]': 'delta0'})
    flightdata = flightdata.rename(columns={'ctrl_1': 'uIn1', 'ctrl_2': 'uIn2', 'ctrl_3': 'uIn3', 'ctrl_4': 'uIn4'})

    flightdata = flightdata.drop(columns=['time_boot_ms', 'status', 'nooutputs',
                                          'ctrl_5', 'ctrl_6', 'ctrl_7', 'ctrl_8', ' nsh[1]'])

    flightdata['t'] = ((flightdata['t'].values - min(flightdata['t'].values)) / 10e5)

    # rename some columns to more intuitive names, added bias from calibration to current and voltage
    # calculate motor power

    for i in range(1, 9):
        flightdata = flightdata.rename(columns={f'U1{i}': f'U{i}',
                                                f'I1{i}': f'I{i}', f'omega{i}': f'rpm{i}', f'delta_{i}': f'uOut{i}'})
        # fd[f'U{i}'] = fd[f'U{i}'] - motorVoltageBias[i-1]
        # fd[f'I{i}'] = fd[f'I{i}'] - motorCurrentBias[i-1]
        # fd[f'pMo{i}'] = fd[f'U{i}'] * fd[f'I{i}']

        flightdata = flightdata.drop(columns=[f'thr_in{i}'])
        flightdata = flightdata.drop(columns=[f'thr_out{i}'])
        flightdata = flightdata.drop(columns=[f'pwm_{i}'])
        flightdata = flightdata.drop(columns=[f'output{i}'])

    return flightdata


def apply_motor_calibration(flightdata):

    motorCurrentBias = np.array([6.36, 2.10, -2.27, 0.09, 6.22, 3.01, -2.66, -1.55])
    motorVoltageBias = np.array([0.07, 0.25, 0.24, 0.24, 0.17, 0.46, -0.04, 0.14])  # bias from ground test
    additionalBias = np.array([0, 0, 0.04, 0, 0, 0, 0, -0.04])  # bias through aggregated flight data

    for i in range(1, 9):
        flightdata[f'U{i}'] = flightdata[f'U{i}'] - motorVoltageBias[i - 1]
        flightdata[f'I{i}'] = flightdata[f'I{i}'] - motorCurrentBias[i - 1]

    return flightdata



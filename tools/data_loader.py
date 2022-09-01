""" Some helper functions to get data from firefly_df """

import numpy as np


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


def moving_average(in_dict, window=5):
    """ Calculates the moving average of the values for every dictionary"""
    out_dict = {}

    # do moving average for numpy array
    if type(in_dict) is np.ndarray:
        return np.convolve(in_dict, np.ones(window), 'same') / window

    for motor in in_dict:
        out_dict[motor] = np.convolve(in_dict[motor], np.ones(window), 'same') / window

    return out_dict


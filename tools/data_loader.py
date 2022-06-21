""" Some helper functions to get data from firefly_df """


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


def get_power(firefly_df):

    current = get_current(firefly_df)
    voltage = get_voltage(firefly_df)

    motor_power = {'11': voltage['11'] * current['11'], '16': voltage['16'] * current['16'],
                   '12': voltage['12'] * current['12'], '15': voltage['15'] * current['15'],
                   '13': voltage['13'] * current['13'], '18': voltage['18'] * current['18'],
                   '14': voltage['14'] * current['14'], '17': voltage['17'] * current['17']}

    return motor_power

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from firefly_preprocessing import get_dfs
from scipy.fft import fft, fftfreq
from scipy.fft import rfft, rfftfreq
import time

"""     Start data analysis  """
data_directory = ['../flight_data/2021-12-10_hangar', '../flight_data/2022-01-24_hangar',
                  '../flight_data/2022-01-24_hangar']

ulg_number = [109, 143, 145]
firefly_number = [54, 5, 9]

for i in range(2, 3):
    firefly_df, arm_df, ulg_df = get_dfs(data_directory[i],
                                         ulg_number[i], firefly_number[i])

    # x-Position: ulg_df['ulg_pv_df']['x']
    x = ulg_df['ulg_pv_df']['x']
    y = ulg_df['ulg_pv_df']['y']
    z = ulg_df['ulg_pv_df']['z']

    # keys to access esc voltage and current
    voltage_esc_key = ['esc11_voltage', 'esc12_voltage', 'esc13_voltage', 'esc14_voltage', 'esc15_voltage',
                       'esc16_voltage', 'esc17_voltage', 'esc18_voltage']

    current_esc_key = ['esc11_current', 'esc12_current', 'esc13_current', 'esc14_current', 'esc15_current',
                       'esc16_current', 'esc17_current', 'esc18_current']

    #voltage_esc_key = ['esc11_voltage']
    #current_esc_key = ['esc11_current']
    rpm_esc_key = ['esc_rpm']
    fig1, ax1 = plt.subplots()  # ground track plot
    fig2, ax2 = plt.subplots()  # altitude plot
    fig3, ax3 = plt.subplots()  # voltage plot
    fig4, ax4 = plt.subplots()  # current plot

    # esc voltage and current
    title_begin = 'Flight {}-{}'.format(ulg_number[i], firefly_number[i])

    ax1.scatter(x, y)
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.grid()
    fig1.suptitle(title_begin + ': ground track')

    ax2.plot(abs(z))
    ax2.grid()
    fig2.suptitle(title_begin + ': altitude')

    min_voltage = 30
    max_voltage = 0
    min_current = 10
    max_current = 0

    time_index = firefly_df.index.values.reshape(-1, 1)

    for volt_key, cur_key in zip(voltage_esc_key, current_esc_key):
        voltage = firefly_df[volt_key]
        current = firefly_df[cur_key]
        min_voltage = min(voltage) if (min(voltage) < min_voltage) else min_voltage
        max_voltage = max(voltage) if (max(voltage) > max_voltage) else max_voltage
        min_current = min(current) if (min(current) < min_current) else min_current
        max_current = max(current) if (max(current) > max_current) else max_current
        ax3.plot(voltage)
        ax4.plot(current)

    title = title_begin + ': Voltage over Time'
    ax3.set_ylim(min(voltage)-1, max(voltage)+1)
    ax3.set_xlim(min(firefly_df.index), max(firefly_df.index))
    ax3.set_xlabel('time [s]')
    ax3.set_ylabel('voltage [V]')
    ax3.grid()
    fig3.suptitle(title)
    fig3.show()
    time.sleep(1)

    title = title_begin + ': Current over Time'
    ax4.set_ylim(min(current) - 1, max(current) + 1)
    # ax4.set_xlim(min(firefly_df.index), max(firefly_df.index))
    ax4.set_xlabel('time [s]')
    ax4.set_ylabel('current [A]')
    ax4.grid()
    fig4.suptitle(title)
    # fig4.show()

    # fig1.show()
    # fig2.show()



# linear regression

regr = linear_model.LinearRegression()
time = firefly_df.index.values.reshape(-1, 1)
current = firefly_df['esc11_current'].values.reshape(-1, 1)
regr.fit(time, current)
pred_vals = regr.predict(time)

ax4.plot(time, pred_vals, 'r')

# removed linear part to get a zero centered signal
current_corrected = firefly_df['esc11_current'].values - pred_vals.reshape(-1)

# fourier analysis
n_samples = current_corrected.size
rfft_vals = rfft(current_corrected)
rfreq = rfftfreq(n_samples, d=1/10)
# ax6.plot(freq, np.abs(fft_vals))

fig4.show()
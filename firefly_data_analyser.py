import matplotlib.pyplot as plt
import numpy as np

from firefly_preprocessing import get_dfs


"""     Start data analysis  """
data_directory = ['../flight_data/2021-12-10_hangar', '../flight_data/2022-01-24_hangar',
                  '../flight_data/2022-01-24_hangar']
ulg_number = [109, 143, 145]

firefly_number = [54, 5, 9]

for i in range(3):
    firefly_df, arm_df, ulg_df = get_dfs(data_directory[i],
                                         ulg_number[i], firefly_number[i])

    # plot groundtrack

    # x-Position: ulg_df['ulg_pv_df']['x']
    x = ulg_df['ulg_pv_df']['x']
    y = ulg_df['ulg_pv_df']['y']
    z = ulg_df['ulg_pv_df']['z']

    title = 'Flight {}-{}'.format(ulg_number[i], firefly_number[i])
    fig1, ax1 = plt.subplots()
    ax1.scatter(x, y)
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)

    fig1.suptitle(title)
    ax1.grid()
    fig1.show()

    fig2, ax2 = plt.subplots()
    ax2.plot(abs(z))

    fig2.suptitle(title)
    ax2.grid()
    fig2.show()

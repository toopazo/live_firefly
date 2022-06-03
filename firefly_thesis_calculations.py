import math
import numpy as np

density = 1.225
weight = 15 * 9.8
rpm_to_rads = math.pi / 30

rotor_radius = 0.311
rotor_area = math.pi * rotor_radius ** 2

rotor_speed_rpm = 2500
rotor_speed = rotor_speed_rpm * rpm_to_rads
print(f'rotor angvel {rotor_speed} rad/s')
print(f'rotor angvel {rotor_speed_rpm / 60} rev/sec')
print(f'rotor area {rotor_area} m2')


vehicle_angvel = (5 * math.pi / 180)
print(f'Typical vehicle angvel is about {vehicle_angvel * 180 / math.pi} deg/s')
print(f'Typical vehicle angvel is about {vehicle_angvel} rad/s')
# print(f'Typical rotor   angvel is about {rotor_speed * 180 / math.pi} deg/s')
print(f'Typical rotor   angvel is about {rotor_speed} rad/s')
# rotor_vel_due_to_rotation = vehicle_angvel * 0.5
# print(f'Typical rotor displacement due to a 5 deg/s rotation is about '
#       f'{rotor_vel_due_to_rotation} m/s')
vehicle_to_rotor_angvel = vehicle_angvel / rotor_speed
print(f'Typical vehicle to rotor angvel is about {vehicle_to_rotor_angvel}')
rotor_to_vehicle_angvel = rotor_speed / vehicle_angvel
print(f'Typical rotor to vehicle angvel is about {rotor_to_vehicle_angvel}')

total_area = 4 * rotor_area
vind_h = math.sqrt(weight / (2 * density * total_area))
power_h = weight * vind_h

print(f'total area {total_area} m2')
print(f'vh {vind_h} m/s')
print(f'power_h {power_h} W')


battery_soc = 0.85
battery_4s_voltage = (4 * 4.2)
rotor_voltage = battery_4s_voltage * 2 * battery_soc
rotor_current = 6
total_current = 8 * rotor_current
power_elec = rotor_voltage * total_current

print(f'rotor_voltage {rotor_voltage} V')
# print(f'rotor_current {rotor_current} A')
print(f'power_elec {power_elec} W')


###
m38_delta_rpm_mean = [0, 0, 0, -679.0, -604.0, -501.0, -400.0, -305.0, -196.0, -88.0, -5.0, 85.0, 204.0, 307.0, 383.0, 493.0, 599.0, 706.0, 792.0, 867.0]
m38_pow_esc_mean = [0, 0, 0, 313.0, 313.0, 309.0, 308.0, 306.0, 317.0, 323.0, 323.0, 323.0, 327.0, 332.0, 331.0, 332.0, 342.0, 351.0, 362.0, 366.0]
m38_pow_esc_std = [0, 0, 0, 10.0, 10.0, 8.0, 8.0, 8.0, 9.0, 10.0, 10.0, 10.0, 9.0, 9.0, 7.0, 10.0, 10.0, 11.0, 8.0, 6.0]

m47_delta_rpm_mean = [-964.0, -911.0, -784.0, -692.0, -602.0, -502.0, -406.0, -294.0, -194.0, -99.0, -7.0, 124.0, 193.0, 311.0, 388.0, 479.0, 600.0, 0, 0, 0]
m47_pow_esc_mean = [312.0, 321.0, 316.0, 315.0, 311.0, 312.0, 313.0, 311.0, 311.0, 315.0, 315.0, 322.0, 325.0, 332.0, 329.0, 338.0, 346.0, 0, 0, 0,]
m47_pow_esc_std = [7.0, 11.0, 15.0, 10.0, 14.0, 11.0, 11.0, 9.0, 12.0, 15.0, 8.0, 7.0, 8.0, 9.0, 8.0, 8.0, 4.0, 0, 0, 0, ]

m38_pm_ref = 323.0
m47_pm_ref = 315
for m38_dm, m38_pm, m38_ps, m47_dm, m47_pm, m47_ps in zip(
        m38_delta_rpm_mean, m38_pow_esc_mean, m38_pow_esc_std,
        m47_delta_rpm_mean, m47_pow_esc_mean, m47_pow_esc_std):
    m38_perc_red = np.around((m38_pm - m38_pm_ref) / m38_pm_ref * 100, 1)
    m47_perc_red = np.around((m47_pm - m47_pm_ref) / m47_pm_ref * 100, 1)
    if (np.abs(m38_perc_red) >= 100) or (np.abs(m47_perc_red) >= 100):
        continue
    print(f'{m38_dm} & {m38_pm} ({m38_perc_red}\%) & {m38_ps} & '
          f'{m47_dm} & {m47_pm} ({m47_perc_red}\%)  & {m47_ps} ')


###
#     tot_pow_esc_mean  tot_pow_esc_std  delta_cmd  net_delta_rpm_mean
# 0        1307.524441        57.576754        0.0          354.503244
# 1        1335.519629        20.759529        0.1          436.919984
# 2        1363.852342        11.766858        0.2          267.102212
# 3        1385.693570        21.870678        0.3          360.181565
# 4        1424.817804        18.606424        0.4          665.763427
# 5        1447.643694        16.313950        0.5          632.997748
# 6        1418.580446        16.995123        0.4          886.871103
# 7        1386.237925        15.229734        0.3          683.352528
# 8        1364.942035        26.878004        0.2          328.170950
# 9        1346.240730        20.892418        0.1          584.785125
# 10       1309.412191        12.100424        0.0          768.814358
# 11       1303.944663        13.801909       -0.1          626.915980
# 12       1298.020419        22.405004       -0.2          697.252012
# 13       1286.000242        24.979244       -0.3          301.320905
# 14       1296.275117        32.411343       -0.4           30.011640
# 15       1294.590648        26.445665       -0.5          562.748989
# 16       1291.712508        23.684581       -0.4          503.846551
# 17       1278.737757        15.685279       -0.3          508.934255
# 18       1300.305605        18.022327       -0.2         1068.729920
# 19       1304.546272        29.239117       -0.1          335.123320

import matplotlib.pyplot as plt

w_inches = 10
figsize = (w_inches, w_inches / 4)

ulg_log_145_delta_cmd = np.array([
    0, -0.05, 0, -0.05,  0, -0.05, 0, -0.05, -0.1, -0.15, -0.2, -0.25, -0.3,
    -0.35, -0.3, -0.35, -0.3, -0.35, -0.4, -0.35, -0.4, -0.45, -0.5, -0.45,
    -0.5, -0.55, -0.6, -0.55, -0.6, -0.65, -0.7, -0.65, -0.7])
# tot_pow_esc_mean
ulg_log_145_tot_pow = np.array([1414.44373569, 1398.51623179, 1415.22199992,
                       1410.91696694, 1419.42714202, 1413.99120803,
                       1425.33439067, 1414.43738224, 1410.60157307,
                       1397.33471458, 1389.7281779, 1385.35163315,
                       1387.16224831, 1395.77364191, 1377.72149066,
                       1386.58713441, 1378.64657164, 1387.3597303,
                       1391.29878348, 1387.73131848, 1395.3574814,
                       1393.5150617, 1391.77454265, 1381.48959918,
                       1382.21352907, 1386.81091386, 1380.59104186,
                       1381.53794623, 1396.33441237, 1392.61932162,
                       1401.80728668, 1390.50247613, 1391.45221418])

ulg_log_144_delta_cmd = np.array([0, -0.05, -0.1, -0.15, -0.2, -0.25, -0.3, -0.35, -0.4,
                         -0.45, -0.5, -0.55, -0.6])
# tot_pow_esc_mean
ulg_log_144_tot_pow = np.array([1400.25628982, 1399.42001841, 1390.03114354,
                       1394.35212342, 1370.84680504, 1373.16310115,
                       1373.1932649, 1372.24043305, 1376.02384622,
                       1375.28086576, 1375.14184284, 1384.68298641,
                       1388.07146426])

ulg_log_143_delta_cmd = np.array([+0.00, +0.05, +0.00, -0.05, -0.10, -0.15, -0.10, -0.15,
                         -0.20, -0.25, -0.20, -0.25, -0.20, -0.25, -0.30, -0.35,
                         -0.40])
# tot_pow_esc_mean
ulg_log_143_tot_pow = np.array([1504.2082615, 1501.59715112, 1491.37815443,
                       1490.50744434, 1482.56954951, 1479.42781267,
                       1472.02006445, 1476.12773746, 1477.00207641,
                       1481.29447131, 1477.7721682, 1482.47961426,
                       1491.837233, 1487.89054935, 1494.86196051,
                       1498.92217015])

ulg_log_137_delta_cmd = np.array([-0.05, 0, -0.05, 0, -0.05, -0.1, -0.15, -0.1, -0.15,
                         -0.1, -0.15])
# tot_pow_esc_mean
ulg_log_137_tot_pow = np.array([1330.47519274, 1311.83709638, 1322.97241779,
                       1330.36865465, 1318.70293763, 1306.39349621,
                       1310.51044018, 1326.05269311, 1305.32200309,
                       1315.34173618, 1307.11957143])

######################
fig, ax1 = plt.subplots(1, 1, figsize=figsize)

ax1.grid(True)
ax1.set_ylabel(r'$ \Delta_0 $ trajectories')
ax1.plot(ulg_log_145_delta_cmd, label='', marker='.')
ax1.plot(ulg_log_144_delta_cmd, label='', marker='.')
ax1.plot(ulg_log_143_delta_cmd, label='', marker='.')
ax1.plot(ulg_log_137_delta_cmd, label='', marker='.')
# ax1.plot(m38_pow_ars_mean, label='$P_{3} + P_{8}$ ars', marker='.')
# ax1.legend(ncol=4, loc='upper right', bbox_to_anchor=(1.01, 1.18),
#           framealpha=1.0)
ax1.set_xlabel("Steps")

file_path = '/home/tzo4/Dropbox/tomas/pennState_avia/firefly_logBook/' \
            '2022-01-20_database/delta_0_trajectories.png'
print(f'Saving file {file_path} ..')
plt.savefig(file_path, bbox_inches='tight')
plt.close(fig)

######################
fig, ax1 = plt.subplots(1, 1, figsize=figsize)

ax1.grid(True)
ax1.set_ylabel(r'Change in power, W')
ax1.plot(ulg_log_145_tot_pow - ulg_log_145_tot_pow[0], label='', marker='.')
ax1.plot(ulg_log_144_tot_pow - ulg_log_144_tot_pow[0], label='', marker='.')
ax1.plot(ulg_log_143_tot_pow - ulg_log_143_tot_pow[0], label='', marker='.')
ax1.plot(ulg_log_137_tot_pow - ulg_log_137_tot_pow[0], label='', marker='.')
# ax1.plot(m38_pow_ars_mean, label='$P_{3} + P_{8}$ ars', marker='.')
# ax1.legend(ncol=4, loc='upper right', bbox_to_anchor=(1.01, 1.18),
#           framealpha=1.0)
ax1.set_xlabel("Steps")
ax1.set_yticks([-40, -30, -20, -10, 0, 10, 20])

file_path = '/home/tzo4/Dropbox/tomas/pennState_avia/firefly_logBook/' \
            '2022-01-20_database/tot_pow_trajectories.png'
print(f'Saving file {file_path} ..')
plt.savefig(file_path, bbox_inches='tight')
plt.close(fig)
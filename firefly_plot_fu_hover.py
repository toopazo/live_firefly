# import argparse
# import copy
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import pandas
import numpy as np
import scipy.stats as stats
# import plotly.graph_objects as go
import plotly.express as px

# from toopazo_tools.file_folder import FileFolderTools as FFTools
from firefly_parse_fu import FUParser, UlgParserTools as UlgPT
from firefly_plot_fu_base import FUPlotBase
from firefly_parse_keys import FireflyDfKeys, FireflyDfKeysMi, UlgDictKeys
from firefly_parse_keys import ArmDfKeys, ArmDfKeysMi
from firefly_parse_keys import UlgInDfKeys, UlgOutDfKeys, UlgPvDfKeys, \
    UlgAngvelDf, UlgAngaccDf, UlgAttDf, UlgAccelDf


class FUPlotHover(FUPlotBase):
    def __init__(self, bdir):
        super().__init__(bdir)

    def norm_vs_time(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        # fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.figsize)
        # fig.suptitle(f'{file_tag}: Variables likely affecting hover power')
        fig, ax = plt.subplots(4, 2, figsize=self.figsize)
        (ax1, ax2, ax3, ax4) = ax[:, 0]
        (ax1t, ax2t, ax3t, ax4t) = ax[:, 1]

        _ = firefly_df

        # ulg_angvel_df = ulg_dict[UlgDictKeys.ulg_angvel_df]
        # ulg_angvelsp_df = ulg_dict[UlgDictKeys.ulg_angvelsp_df]
        ulg_in_df = ulg_dict[UlgDictKeys.ulg_in_df]
        ulg_out_df = ulg_dict[UlgDictKeys.ulg_out_df]
        ulg_pv_df = ulg_dict[UlgDictKeys.ulg_pv_df]
        ulg_angvel_df = ulg_dict[UlgDictKeys.ulg_angvel_df]
        ulg_accel_df = ulg_dict[UlgDictKeys.ulg_accel_df]
        ulg_angacc_df = ulg_dict[UlgDictKeys.ulg_angacc_df]

        smooth = False
        if smooth:
            # fig.suptitle(f'{file_tag}: L2-norm of smoothed signals')
            fig.suptitle('Norm of smoothed signals')
            vel_norm = pandas.DataFrame(
                index=ulg_pv_df.index, data={'data': savgol_filter(
                    ulg_pv_df[UlgPvDfKeys.vel_norm].values, 21, 3)}
            )['data']
            pqr_norm = pandas.DataFrame(
                index=ulg_angvel_df.index, data={'data': savgol_filter(
                    ulg_angvel_df[UlgAngvelDf.pqr_norm].values, 21, 3)}
            )['data']
            acc_norm = pandas.DataFrame(
                index=ulg_accel_df.index, data={'data': savgol_filter(
                    ulg_accel_df[UlgAccelDf.acc_norm].values, 21, 3)}
            )['data']
            angacc_norm = pandas.DataFrame(
                index=ulg_angacc_df.index, data={'data': savgol_filter(
                    ulg_angacc_df[UlgAngaccDf.angacc_norm].values, 21, 3)}
            )['data']
        else:
            vel_norm = ulg_pv_df[UlgPvDfKeys.vel_norm]
            pqr_norm = ulg_angvel_df[UlgAngvelDf.pqr_norm]
            acc_norm = ulg_accel_df[UlgAccelDf.acc_norm]
            angacc_norm = ulg_angacc_df[UlgAngaccDf.angacc_norm]

        # accel_measurment = f = T = accel - G => accel = T + G = 0 => T = -9.8
        # acc_norm = acc_norm - np.mean(acc_norm)
        acc_norm = acc_norm - 9.8

        # mask = FUParser.get_hover_mask(
        #     norm_arr=[vel_norm, pqr_norm],
        #     ulg_df_arr=[ulg_pv_df, ulg_in_df, ulg_out_df],
        #     arm_df=arm_df)

        ax1.grid(True)
        ax1.set_ylabel('Vel, m/s')
        # ax1.scatter(vel_norm[mask].index, vel_norm[mask].values*0,
        #             color='red', s=10)
        ax1.plot(vel_norm)
        ax1.axes.xaxis.set_ticklabels([])

        ax2.grid(True)
        ax2.set_ylabel('PQR, deg/s')
        ax2.plot(pqr_norm)
        ax2.axes.xaxis.set_ticklabels([])

        ax3.grid(True)
        ax3.set_ylabel('Accel, $m/s^2$')
        ax3.plot(acc_norm)
        ax3.axes.xaxis.set_ticklabels([])

        ax4.grid(True)
        ax4.set_ylabel('Angacc, $deg/s^2$')
        ax4.plot(angacc_norm)
        ax4.set_xlabel("Time, s")

        num_bins = 50

        axt_arr = [ax1t, ax2t, ax3t, ax4t]
        ymax_arr = [1, 10, 2, 400]
        data_arr = [vel_norm, pqr_norm, acc_norm, angacc_norm]
        for axt, ymax, data in zip(axt_arr, ymax_arr, data_arr):
            # axt.set_ylabel("$| f(x) |$")
            axt.set_ylabel("$ pdf $")

            mu, sigma = stats.norm.fit(data)
            data = (data - mu) / sigma
            ax_lbl = f'$\mu$ {round(mu, 2)} $\sigma$ {round(sigma, 2)}'
            _, bins, _ = axt.hist(
                data, bins=num_bins, density=True, alpha=0.5, label=ax_lbl)
            axt.legend(ncol=4, loc='upper right', bbox_to_anchor=(1.0, 1.2),
                       framealpha=1.0)
            axt.grid(True)
            if ymax == 400:
                ax_ylabel = r'$ (x -\mu) \  / \ \sigma $'
                axt.set_xlabel(ax_ylabel)
            else:
                axt.axes.xaxis.set_ticklabels([])

        # ymin = -0.5
        # ymax = +0.5
        FUPlotHover.set_axes_limits(
            [ax1, ax2, ax3, ax4],
            [[], [], [], []],
            [[0, 1], [0, 10], [0, 2], [0, 400]]
        )

        xmin = -3
        xmax = +3
        FUPlotHover.set_axes_limits(
            [ax1t, ax2t, ax3t, ax4t],
            [[xmin, xmax], [xmin, xmax], [xmin, xmax], [xmin, xmax]],
            [[0, 1], [0, 1], [0, 1], [0, 1]]
        )

        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

    def ver_thr(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.figsize)
        # fig.suptitle(f'{file_tag}: Variables likely affecting hover power')
        _ = firefly_df, ulg_dict
        var = 'thr'
        # var = 'rpm'
        # var = 'cur'
        FUPlotHover.rate_thr_rpm_cur(
            arm_df, plot_variable=var, ax_arr=[ax1, ax2, ax3, ax4])
        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

    def ver_rpm(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.figsize)
        # fig.suptitle(f'{file_tag}: Variables likely affecting hover power')
        _ = firefly_df, ulg_dict
        # var = 'thr'
        var = 'rpm'
        # var = 'cur'
        FUPlotHover.rate_thr_rpm_cur(
            arm_df, plot_variable=var, ax_arr=[ax1, ax2, ax3, ax4])
        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

    def ver_cur(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.figsize)
        # fig.suptitle(f'{file_tag}: Variables likely affecting hover power')
        _ = firefly_df, ulg_dict
        # var = 'thr'
        # var = 'rpm'
        var = 'cur'
        FUPlotHover.rate_thr_rpm_cur(
            arm_df, plot_variable=var, ax_arr=[ax1, ax2, ax3, ax4])
        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

    @staticmethod
    def rate_thr_rpm_cur(arm_df, plot_variable, ax_arr):
        [ax1, ax2, ax3, ax4] = ax_arr

        mi_var_dict = {}
        mi_var = None
        ymin = None
        ymax = None
        for mi in range(1, 9):
            if plot_variable == 'thr':
                ymin = -30
                ymax = +30
                mi_var = arm_df[ArmDfKeysMi(mi).rate_thr]
            if plot_variable == 'rpm':
                ymin = -50
                ymax = +50
                mi_var = arm_df[ArmDfKeysMi(mi).rate_rpm]
            if plot_variable == 'cur':
                ymin = -1
                ymax = +1
                mi_var = arm_df[ArmDfKeysMi(mi).rate_cur]
            mi_var_dict[f'm{mi}'] = mi_var

        key_m_upper = 'm1'
        key_m_lower = 'm6'
        ylbl = f"{plot_variable} rate {key_m_upper}{key_m_lower[1]}"
        ax1.grid(True)
        ax1.set_ylabel(ylbl)
        ax1.plot(mi_var_dict[key_m_upper], alpha=0.5, label=key_m_upper)
        ax1.plot(mi_var_dict[key_m_lower], alpha=0.5, label=key_m_lower)
        ax1.axes.xaxis.set_ticklabels([])
        ax1.legend(ncol=2, loc='upper left')

        key_m_upper = 'm2'
        key_m_lower = 'm5'
        ylbl = f"{plot_variable} rate {key_m_upper}{key_m_lower[1]}"
        ax2.grid(True)
        ax2.set_ylabel(ylbl)
        ax2.plot(mi_var_dict[key_m_upper], alpha=0.5, label=key_m_upper)
        ax2.plot(mi_var_dict[key_m_lower], alpha=0.5, label=key_m_lower)
        ax2.axes.xaxis.set_ticklabels([])
        ax2.legend(ncol=2, loc='upper left')

        key_m_upper = 'm3'
        key_m_lower = 'm8'
        ylbl = f"{plot_variable} rate {key_m_upper}{key_m_lower[1]}"
        ax3.grid(True)
        ax3.set_ylabel(ylbl)
        ax3.plot(mi_var_dict[key_m_upper], alpha=0.5, label=key_m_upper)
        ax3.plot(mi_var_dict[key_m_lower], alpha=0.5, label=key_m_lower)
        ax3.axes.xaxis.set_ticklabels([])
        ax3.legend(ncol=2, loc='upper left')

        key_m_upper = 'm4'
        key_m_lower = 'm7'
        ylbl = f"{plot_variable} rate {key_m_upper}{key_m_lower[1]}"
        ax4.grid(True)
        ax4.set_ylabel(ylbl)
        ax4.plot(mi_var_dict[key_m_upper], alpha=0.5, label=key_m_upper)
        ax4.plot(mi_var_dict[key_m_lower], alpha=0.5, label=key_m_lower)
        ax4.set_xlabel("Time, s")
        ax4.legend(ncol=2, loc='upper left')

        FUPlotHover.set_axes_limits(
            [ax1, ax2, ax3, ax4],
            [[], [], [], []],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]]
        )

    def thr_stats(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        fig, ax = plt.subplots(4, 2, figsize=self.figsize)
        _ = arm_df, ulg_dict
        var = 'thr'
        # var = 'rpm'
        # var = 'cur'
        FUPlotHover.thr_rpm_cur_vol_stats(firefly_df, plot_variable=var, axes=ax)

        (ax1, ax2, ax3, ax4) = ax[:, 0]
        ymin = -100
        ymax = +100
        FUPlotHover.set_axes_limits(
            [ax1, ax2, ax3, ax4],
            [[], [], [], []],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]])

        (ax1t, ax2t, ax3t, ax4t) = ax[:, 1]
        xmin = -3
        xmax = +3
        FUPlotHover.set_axes_limits(
            [ax1t, ax2t, ax3t, ax4t],
            [[xmin, xmax], [xmin, xmax], [xmin, xmax], [xmin, xmax]],
            [[0, 1], [0, 1], [0, 1], [0, 1]]
        )

        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

    def rpm_stats(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        fig, ax = plt.subplots(4, 2, figsize=self.figsize)
        _ = arm_df, ulg_dict
        # var = 'thr'
        var = 'rpm'
        # var = 'cur'
        FUPlotHover.thr_rpm_cur_vol_stats(firefly_df, plot_variable=var, axes=ax)

        (ax1, ax2, ax3, ax4) = ax[:, 0]
        ymin = -150
        ymax = +150
        FUPlotHover.set_axes_limits(
            [ax1, ax2, ax3, ax4],
            [[], [], [], []],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]])

        (ax1t, ax2t, ax3t, ax4t) = ax[:, 1]
        xmin = -3
        xmax = +3
        FUPlotHover.set_axes_limits(
            [ax1t, ax2t, ax3t, ax4t],
            [[xmin, xmax], [xmin, xmax], [xmin, xmax], [xmin, xmax]],
            [[0, 1], [0, 1], [0, 1], [0, 1]]
        )

        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

    def cur_stats(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        fig, ax = plt.subplots(4, 2, figsize=self.figsize)
        _ = arm_df, ulg_dict
        # var = 'thr'
        # var = 'rpm'
        var = 'cur'
        FUPlotHover.thr_rpm_cur_vol_stats(firefly_df, plot_variable=var, axes=ax)

        (ax1, ax2, ax3, ax4) = ax[:, 0]
        ymin = -2
        ymax = +2
        FUPlotHover.set_axes_limits(
            [ax1, ax2, ax3, ax4],
            [[], [], [], []],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]])

        (ax1t, ax2t, ax3t, ax4t) = ax[:, 1]
        xmin = -3
        xmax = +3
        FUPlotHover.set_axes_limits(
            [ax1t, ax2t, ax3t, ax4t],
            [[xmin, xmax], [xmin, xmax], [xmin, xmax], [xmin, xmax]],
            [[0, 1], [0, 1], [0, 1], [0, 1]]
        )

        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

    def vol_stats(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        fig, ax = plt.subplots(4, 2, figsize=self.figsize)
        _ = arm_df, ulg_dict
        # var = 'thr'
        # var = 'rpm'
        # var = 'cur'
        var = 'vol'
        FUPlotHover.thr_rpm_cur_vol_stats(firefly_df, plot_variable=var,
                                          axes=ax)

        (ax1, ax2, ax3, ax4) = ax[:, 0]
        ymin = -0.5
        ymax = +0.5
        FUPlotHover.set_axes_limits(
            [ax1, ax2, ax3, ax4],
            [[], [], [], []],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]])

        (ax1t, ax2t, ax3t, ax4t) = ax[:, 1]
        xmin = -3
        xmax = +3
        FUPlotHover.set_axes_limits(
            [ax1t, ax2t, ax3t, ax4t],
            [[xmin, xmax], [xmin, xmax], [xmin, xmax], [xmin, xmax]],
            [[0, 1], [0, 1], [0, 1], [0, 1]]
        )

        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

    @staticmethod
    def thr_rpm_cur_vol_stats(firefly_df, plot_variable, axes):
        # [ax1, ax2, ax3, ax4] = ax_arr
        (ax1, ax2, ax3, ax4) = axes[:, 0]
        (ax1t, ax2t, ax3t, ax4t) = axes[:, 1]

        mi_var_dict = {}
        units = None
        ax_ylabel = None
        for mi in range(1, 9):
            if plot_variable == 'thr':
                units = r'$\mu s$'
                ax_ylabel = r"$\delta$ - $\overline{\delta}$, %s" % units
                mi_var = firefly_df[FireflyDfKeysMi(mi).thr]
            elif plot_variable == 'rpm':
                units = r'rpm'
                ax_ylabel = r"$\Omega$ - $\overline{\Omega}$, %s" % units
                mi_var = firefly_df[FireflyDfKeysMi(mi).rpm]
            elif plot_variable == 'cur':
                units = r'$A$'
                ax_ylabel = r"$I$ - $\overline{I}$, %s" % units
                mi_var = firefly_df[FireflyDfKeysMi(mi).cur]
                # units = r'$\frac{A}{10}$'
                # ax_xlabel = r"$I$ - $\overline{I}$, %s" % units
                # mi_var = firefly_df[FireflyDfKeysMi(mi).cur] * 10
                # units = r'$mA$'
                # ax_xlabel = r"$I$ - $\overline{I}$, %s" % units
                # mi_var = firefly_df[FireflyDfKeysMi(mi).cur] * 1000
            elif plot_variable == 'vol':
                units = r'V'
                ax_ylabel = r"$V$ - $\overline{V}$, %s" % units
                mi_var = firefly_df[FireflyDfKeysMi(mi).vol]
            else:
                raise RuntimeError
            mi_var_dict[f'm{mi}'] = mi_var

        ax_arr = [ax1, ax2, ax3, ax4]
        kmu_arr = ['m1', 'm2', 'm3', 'm4']
        kml_arr = ['m6', 'm5', 'm8', 'm7']
        for ax, key_m_upper, key_m_lower in zip(ax_arr, kmu_arr, kml_arr):
            ax.grid(True)
            # ax_ylabel = f'Arm {key_m_upper[1]}, {units}'
            ax.set_ylabel(ax_ylabel)

            # ax.plot(data, alpha=0.5, label=key_m_upper)
            data = mi_var_dict[key_m_upper]
            val_mean = np.mean(data.values)
            ax.plot(data - val_mean, alpha=0.5)

            # ax.plot(mi_var_dict[key_m_lower], alpha=0.5, label=key_m_lower)
            data = mi_var_dict[key_m_lower]
            val_mean = np.mean(data.values)
            ax.plot(data - val_mean, alpha=0.5)

            if key_m_upper == 'm4':
                ax.set_xlabel("Time, s")
            else:
                ax.axes.xaxis.set_ticklabels([])

        num_bins = 50

        axt_arr = [ax1t, ax2t, ax3t, ax4t]
        kmu_arr = ['m1', 'm2', 'm3', 'm4']
        kml_arr = ['m6', 'm5', 'm8', 'm7']
        for axt, key_m_upper, key_m_lower in zip(axt_arr, kmu_arr, kml_arr):
            # axt.set_ylabel("$| f(x) |$")
            axt.set_ylabel("$ pdf $")

            data = mi_var_dict[key_m_upper]
            mu, sigma = stats.norm.fit(data)
            if plot_variable == 'cur' or plot_variable == 'vol':
                mu_lbl = round(mu, 2)
                sigma_lbl = round(sigma, 2)
            else:
                mu_lbl = int(mu)
                sigma_lbl = int(sigma)
            ax_lbl = f'u : $\mu$ {mu_lbl} $\sigma$ {sigma_lbl}'
            data = (data - mu) / sigma
            _, bins, _ = axt.hist(
                data, bins=num_bins, density=True, alpha=0.5, label=ax_lbl)
            # best_fit_line = stats.norm.pdf(bins, loc=mu, scale=sigma)
            best_fit_line = stats.norm.pdf(bins)
            axt.plot(bins, best_fit_line, color='black')

            data = mi_var_dict[key_m_lower]
            mu, sigma = stats.norm.fit(data)
            if plot_variable == 'cur' or plot_variable == 'vol':
                mu_lbl = round(mu, 2)
                sigma_lbl = round(sigma, 2)
            else:
                mu_lbl = int(mu)
                sigma_lbl = int(sigma)
            ax_lbl = f'l : $\mu$ {mu_lbl} $\sigma$ {sigma_lbl}'
            data = (data - mu) / sigma
            _, bins, _ = axt.hist(
                data, bins=num_bins, density=True, alpha=0.5, label=ax_lbl)
            # best_fit_line = stats.norm.pdf(bins, loc=mu, scale=sigma)
            best_fit_line = stats.norm.pdf(bins)
            axt.plot(bins, best_fit_line, color='black')

            axt.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.2),
                       framealpha=1.0)
            axt.grid(True)
            if key_m_upper == 'm4':
                ax_ylabel = r'$ (x -\mu) \  / \ \sigma $'
                axt.set_xlabel(ax_ylabel)
            else:
                axt.axes.xaxis.set_ticklabels([])

    def delta_rpm_mean(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        nshstats_df = FUParser.calculate_nshstats_df(
            firefly_df, arm_df, ulg_dict, file_tag)

        lower_dict = {
            'm16_delta_rpm_mean': [-1000],
            'm25_delta_rpm_mean': [-1000],
            'm38_delta_rpm_mean': [-1000],
            'm47_delta_rpm_mean': [-1000],
            'delta_cmd': [-0.6],
        }
        lower_df = pandas.DataFrame.from_dict(
            data=lower_dict, orient='columns', dtype=None, columns=None)
        nshstats_df = nshstats_df.append(
            lower_df, verify_integrity=True, ignore_index=True)
        upper_dict = {
            'm16_delta_rpm_mean': [1000],
            'm25_delta_rpm_mean': [1000],
            'm38_delta_rpm_mean': [1000],
            'm47_delta_rpm_mean': [1000],
            'delta_cmd': [+0.6],
        }
        upper_df = pandas.DataFrame.from_dict(
            data=upper_dict, orient='columns', dtype=None, columns=None)
        nshstats_df = nshstats_df.append(
            upper_df, verify_integrity=True, ignore_index=True)

        # https://plotly.com/python-api-reference/generated/
        # plotly.express.parallel_coordinates.html
        dim_arr = ['m16_delta_rpm_mean', 'm25_delta_rpm_mean',
                   'm38_delta_rpm_mean', 'm47_delta_rpm_mean', 'delta_cmd']
        fig = px.parallel_coordinates(
            nshstats_df, dimensions=dim_arr, color='delta_cmd', labels={
                'm16_delta_rpm_mean': 'Arm 1, rpm',
                'm25_delta_rpm_mean': 'Arm 2, rpm',
                'm38_delta_rpm_mean': 'Arm 3, rpm',
                'm47_delta_rpm_mean': 'Arm 4, rpm',
                'delta_cmd': 'Delta cmd'},
            color_continuous_scale=[
                (0.00, "white"), (0.01, "white"),
                (0.01, "red"), (0.33, "red"),
                (0.33, "green"), (0.66, "green"),
                (0.66, "blue"), (0.99, "blue"),
                (0.99, "white"), (1.00, "white"),
            ],
            title='Difference in RPM (mean)', width=900, height=450,
        )

        fig.update_layout(
            font=dict(
                family="Time New Roman",
                size=18,
                color="Black"
            )
        )

        fp = self.save_current_plot(
            file_tag, tag_arr=tag_arr, sep="_", ext='.png')
        fig.write_image(fp)

    def delta_rpm_std(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        nshstats_df = FUParser.calculate_nshstats_df(
            firefly_df, arm_df, ulg_dict, file_tag)

        lower_dict = {
            'm16_delta_rpm_std': [0],
            'm25_delta_rpm_std': [0],
            'm38_delta_rpm_std': [0],
            'm47_delta_rpm_std': [0],
            'delta_cmd': [-0.6],
        }
        lower_df = pandas.DataFrame.from_dict(
            data=lower_dict, orient='columns', dtype=None, columns=None)
        nshstats_df = nshstats_df.append(
            lower_df, verify_integrity=True, ignore_index=True)
        upper_dict = {
            'm16_delta_rpm_std': [150],
            'm25_delta_rpm_std': [150],
            'm38_delta_rpm_std': [150],
            'm47_delta_rpm_std': [150],
            'delta_cmd': [+0.6],
        }
        upper_df = pandas.DataFrame.from_dict(
            data=upper_dict, orient='columns', dtype=None, columns=None)
        nshstats_df = nshstats_df.append(
            upper_df, verify_integrity=True, ignore_index=True)

        # https://plotly.com/python-api-reference/generated/
        # plotly.express.parallel_coordinates.html
        dim_arr = ['m16_delta_rpm_std', 'm25_delta_rpm_std',
                   'm38_delta_rpm_std', 'm47_delta_rpm_std', 'delta_cmd']
        fig = px.parallel_coordinates(
            nshstats_df, dimensions=dim_arr, color='delta_cmd', labels={
                'm16_delta_rpm_std': 'Arm 1, rpm',
                'm25_delta_rpm_std': 'Arm 2, rpm',
                'm38_delta_rpm_std': 'Arm 3, rpm',
                'm47_delta_rpm_std': 'Arm 4, rpm',
                'delta_cmd': 'Delta cmd'},
            color_continuous_scale=[
                (0.00, "white"), (0.01, "white"),
                (0.01, "red"), (0.33, "red"),
                (0.33, "green"), (0.66, "green"),
                (0.66, "blue"), (0.99, "blue"),
                (0.99, "white"), (1.00, "white"),
            ],
            title='Difference in RPM (std)', width=900, height=450,
        )

        fig.update_layout(
            font=dict(
                family="Time New Roman",
                size=18,
                color="Black"
            )
        )

        fp = self.save_current_plot(
            file_tag, tag_arr=tag_arr, sep="_", ext='.png')
        fig.write_image(fp)
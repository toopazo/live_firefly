# import argparse
# import copy
# from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import os
import scipy.stats as stats
import numpy as np
# import plotly.graph_objects as go
import plotly.express as px
import pandas

# from toopazo_tools.file_folder import FileFolderTools as FFTools
from firefly_parse_fu import FUParser, UlgParserTools as UlgPT
from firefly_database import FileTagData
from firefly_parse_keys import FireflyDfKeys, FireflyDfKeysMi, UlgDictKeys
from firefly_parse_keys import ArmDfKeys, ArmDfKeysMi
from firefly_parse_keys import UlgInDfKeys, UlgOutDfKeys, UlgPvDfKeys, \
    UlgAngvelDf, UlgAngaccDf, UlgAttDf, UlgAccelDf


class FUPlotPower:
    def __init__(self, bdir):
        self.logdir = bdir + '/logs'
        self.tmpdir = bdir + '/tmp'
        self.plotdir = bdir + '/plots'
        try:
            if not os.path.isdir(self.logdir):
                os.mkdir(self.logdir)
            if not os.path.isdir(self.tmpdir):
                os.mkdir(self.logdir)
            if not os.path.isdir(self.plotdir):
                os.mkdir(self.logdir)
        except OSError:
            raise RuntimeError(
                'Directories are not present or could not be created')

        self.bdir = bdir
        # w_inches = 6.5
        # w_inches = 7
        w_inches = 10
        self.figsize = (w_inches, w_inches / 2)
        # plt.rcParams.update({'font.size': 10})

    def save_current_plot(self, file_tag, tag_arr, sep, ext):
        file_name = file_tag
        for tag in tag_arr:
            file_name = file_name + sep + str(tag)
        file_path = self.plotdir + f'/' + file_name + ext

        # plt.show()
        print(f'Saving file {file_path} ..')
        plt.savefig(file_path, bbox_inches='tight')
        plt.close(plt.gcf())
        return file_path

    @staticmethod
    def set_axes_limits(axs, xlims, ylims):
        i = 0
        for ax in axs:
            xmin_xmax = xlims[i]
            if len(xmin_xmax) == 2:
                xmin = xmin_xmax[0]
                xmax = xmin_xmax[1]
                if (xmin is not None) and (xmax is not None):
                    ax.set_xlim([xmin, xmax])
            ymin_ymax = ylims[i]
            if len(ymin_ymax) == 2:
                ymin = ymin_ymax[0]
                ymax = ymin_ymax[1]
                if (ymin is not None) and (ymax is not None):
                    ax.set_ylim([ymin, ymax])
            i = i + 1

    def pow_stats(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        # fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.figsize)
        # fig.suptitle(f'{file_tag}')
        fig, ax = plt.subplots(4, 2, figsize=self.figsize)
        (ax1, ax2, ax3, ax4) = ax[:, 0]
        (ax1t, ax2t, ax3t, ax4t) = ax[:, 1]

        m16_pow_esc = arm_df[ArmDfKeys.m16.pow_esc]
        m25_pow_esc = arm_df[ArmDfKeys.m25.pow_esc]
        m38_pow_esc = arm_df[ArmDfKeys.m38.pow_esc]
        m47_pow_esc = arm_df[ArmDfKeys.m47.pow_esc]

        # m16_pow_ars = arm_df[ArmDfKeys.m16.pow_ars']
        # m25_pow_ars = arm_df[ArmDfKeys.m25.pow_ars']
        # m38_pow_ars = arm_df[ArmDfKeys.m38.pow_ars]
        # m47_pow_ars = arm_df[ArmDfKeys.m47.pow_ars]

        ax1.grid(True)
        # ax1.set_ylabel("$P_1$ + $P_6$, W")
        ax1.set_ylabel(r"$P$ - $\overline{P}$, W")
        # ax1.plot(m16_pow_esc, label='m1 + m6', alpha=al)
        val_mean = np.mean(m16_pow_esc.values)
        ax1.plot(m16_pow_esc - val_mean)
        ax1.axes.xaxis.set_ticklabels([])

        ax2.grid(True)
        ax2.set_ylabel(r"$P$ - $\overline{P}$, W")
        # ax2.plot(m25_pow_esc, label='m2 + m5', alpha=al)
        val_mean = np.mean(m25_pow_esc.values)
        ax2.plot(m25_pow_esc - val_mean)
        ax2.axes.xaxis.set_ticklabels([])

        ax3.grid(True)
        ax3.set_ylabel(r"$P$ - $\overline{P}$, W")
        # ax3.plot(m38_pow_esc, label='m3 + m8', alpha=al)
        # ax3.plot(m38_pow_ars, label='m3 + m8 ars', alpha=al)
        val_mean = np.mean(m38_pow_esc.values)
        ax3.plot(m38_pow_esc - val_mean)
        ax3.axes.xaxis.set_ticklabels([])

        ax4.grid(True)
        ax4.set_ylabel(r"$P$ - $\overline{P}$, W")
        # ax4.plot(m47_pow_esc, label='m4 + m7', alpha=al)
        # ax4.plot(m47_pow_ars, label='m4 + m7 ars', alpha=al)
        val_mean = np.mean(m47_pow_esc.values)
        ax4.plot(m47_pow_esc - val_mean)
        ax4.set_xlabel("Time, s")

        num_bins = 30

        flag_arr = [True, True, True, False]
        ax_arr = [ax1t, ax2t, ax3t, ax4t]
        data_arr = [m16_pow_esc.values, m25_pow_esc.values,
                    m38_pow_esc.values, m47_pow_esc.values]
        for flag, axt, data in zip(flag_arr, ax_arr, data_arr):
            val_mean = np.mean(data)
            val_std = np.std(data)
            axt_lbl = f'$\mu$ {round(val_mean, 2)} $\sigma$ {round(val_std, 2)}'

            _, bins, _ = axt.hist(
                data, bins=num_bins, density=True, alpha=0.5, label=axt_lbl)
            mu, sigma = stats.norm.fit(data)
            best_fit_line = stats.norm.pdf(bins, loc=mu, scale=sigma)
            axt.plot(bins, best_fit_line, color='black')
            axt.legend(ncol=4, loc='upper left')
            # axt.set_ylabel("$| f(x) |$")
            axt.set_ylabel("$ pdf $")
            axt.grid(True)
            if flag:
                axt.axes.xaxis.set_ticklabels([])
            else:
                axt.set_xlabel('Power, W')

        ymin = -100
        ymax = +100
        FUPlotPower.set_axes_limits(
            [ax1, ax2, ax3, ax4],
            [[], [], [], []],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]])

        xmin = 200
        xmax = 400
        FUPlotPower.set_axes_limits(
            [ax1t, ax2t, ax3t, ax4t],
            [[xmin, xmax], [xmin, xmax], [xmin, xmax], [xmin, xmax]],
            [[0, 0.04], [0, 0.04], [0, 0.04], [0, 0.04]]
        )

        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

    def pow_residuals(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.figsize)
        # fig.suptitle(f'{file_tag}: Power estimation')
        # fig.suptitle('Power estimation')

        powest_df = FUParser.get_power_estimation(firefly_df, arm_df, ulg_dict)

        m16_res_est1 = np.abs(powest_df['m16_res_est1'])
        m25_res_est1 = np.abs(powest_df['m25_res_est1'])
        m38_res_est1 = np.abs(powest_df['m38_res_est1'])
        m47_res_est1 = np.abs(powest_df['m47_res_est1'])

        m16_res_est2 = np.abs(powest_df['m16_res_est2'])
        m25_res_est2 = np.abs(powest_df['m25_res_est2'])
        m38_res_est2 = np.abs(powest_df['m38_res_est2'])
        m47_res_est2 = np.abs(powest_df['m47_res_est2'])

        al = 0.5
        # ls = ':'

        ax1.grid(True)
        ax1.set_ylabel("Residuals, W")
        ax1.plot(m16_res_est1, label='ca in', alpha=al)
        ax1.plot(m16_res_est2, label='rpm', alpha=al)
        ax1.legend(ncol=2, loc='upper left')
        ax1.axes.xaxis.set_ticklabels([])

        ax2.grid(True)
        ax2.set_ylabel("Residuals, W")
        ax2.plot(m25_res_est1, label='ca in', alpha=al)
        ax2.plot(m25_res_est2, label='rpm', alpha=al)
        ax2.legend(ncol=2, loc='upper left')
        ax2.axes.xaxis.set_ticklabels([])

        ax3.grid(True)
        ax3.set_ylabel("Residuals, W")
        ax3.plot(m38_res_est1, label='ca in', alpha=al)
        ax3.plot(m38_res_est2, label='rpm', alpha=al)
        ax3.legend(ncol=4, loc='upper left')
        ax3.axes.xaxis.set_ticklabels([])

        ax4.grid(True)
        ax4.set_ylabel("Residuals, W")
        ax4.plot(m47_res_est1, label='ca in', alpha=al)
        ax4.plot(m47_res_est2, label='rpm', alpha=al)
        ax4.legend(ncol=4, loc='upper left')
        ax4.set_xlabel("Time, s")

        ymin = -100 * 0
        ymax = +100 * 2
        FUPlotPower.set_axes_limits(
            [ax1, ax2, ax3, ax4],
            [[], [], [], []],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]]
        )

        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

    def pow_vs_delta_rpm(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.figsize)
        # fig.suptitle(f'{file_tag}')

        # m16_pow_ars = arm_df[ArmDfKeys.m16.pow_ars'].values
        # m25_pow_ars = arm_df[ArmDfKeys.m25.pow_ars'].values
        # m38_pow_ars = arm_df[ArmDfKeys.m38.pow_ars].values
        # m47_pow_ars = arm_df[ArmDfKeys.m47.pow_ars].values

        m16_pow_esc = arm_df[ArmDfKeys.m16.pow_esc].values
        m25_pow_esc = arm_df[ArmDfKeys.m25.pow_esc].values
        m38_pow_esc = arm_df[ArmDfKeys.m38.pow_esc].values
        m47_pow_esc = arm_df[ArmDfKeys.m47.pow_esc].values

        m16_delta_rpm = arm_df[ArmDfKeys.m16.delta_rpm].values
        m25_delta_rpm = arm_df[ArmDfKeys.m25.delta_rpm].values
        m38_delta_rpm = arm_df[ArmDfKeys.m38.delta_rpm].values
        m47_delta_rpm = arm_df[ArmDfKeys.m47.delta_rpm].values

        # numpy.polyfit(x, y, deg)
        sorted_delta = np.sort(m16_delta_rpm)
        sorted_power = m16_pow_esc[np.argsort(np.sort(m16_delta_rpm))]
        print(sorted_delta)
        print(sorted_power)
        m16_polyfit = np.polyfit(
            x=sorted_delta,
            y=sorted_power,
            deg=2)
        m16_pol = np.poly1d(m16_polyfit)
        print(f'm16_pol {m16_pol}')

        ax1.grid(True)
        ax1.set_ylabel(r"$P_1$ + $P_6$, W")
        ax1.scatter(x=m16_delta_rpm, y=m16_pow_esc, s=2, alpha=0.5)
        ax1.scatter(x=m16_delta_rpm, y=m16_pol(m16_pow_esc), s=2,
                    alpha=0.5)
        ax1.axes.xaxis.set_ticklabels([])

        ax2.grid(True)
        ax2.set_ylabel(r"$P_2$ + $P_5$, W")
        ax2.scatter(x=m25_delta_rpm, y=m25_pow_esc, s=2, alpha=0.5)
        ax2.axes.xaxis.set_ticklabels([])

        ax3.grid(True)
        ax3.set_ylabel(r"$P_3$ + $P_8$, W")
        ax3.scatter(x=m38_delta_rpm, y=m38_pow_esc,
                    s=2, alpha=0.5, label='esc')
        # ax3.scatter(x=m38_delta_rpm, y=m38_pow_ars,
        #             s=2, alpha=0.5, label='esc')
        # ax3.legend(ncol=4, loc='lower left')
        ax3.axes.xaxis.set_ticklabels([])

        ax4.grid(True)
        ax4.set_ylabel(r"$P_4$ + $P_7$, W")
        ax4.scatter(x=m47_delta_rpm, y=m47_pow_esc,
                    s=2, alpha=0.5, label='esc')
        # ax4.scatter(x=m47_delta_rpm, y=m47_pow_ars,
        #             s=2, alpha=0.5, label='ars')
        # ax4.legend(ncol=4, loc='lower left')
        # ax4.set_xlabel("delta_rpm")
        ax4.set_xlabel(r"$\Delta \Omega$, rpm")

        xmin = -1500
        xmax = +1500
        ymin = 200
        ymax = 500
        FUPlotPower.set_axes_limits(
            [ax1, ax2, ax3, ax4],
            [[xmin, xmax], [xmin, xmax], [xmin, xmax], [xmin, xmax]],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]])

        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

    def pow_vs_eta_rpm(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.figsize)
        # fig.suptitle(f'{file_tag}')

        # m16_pow_ars = arm_df[ArmDfKeys.m16.pow_ars]
        # m25_pow_ars = arm_df[ArmDfKeys.m25.pow_ars]
        # m38_pow_ars = arm_df[ArmDfKeys.m38.pow_ars].values
        # m47_pow_ars = arm_df[ArmDfKeys.m47.pow_ars].values

        m16_pow_esc = arm_df[ArmDfKeys.m16.pow_esc].values
        m25_pow_esc = arm_df[ArmDfKeys.m25.pow_esc].values
        m38_pow_esc = arm_df[ArmDfKeys.m38.pow_esc].values
        m47_pow_esc = arm_df[ArmDfKeys.m47.pow_esc].values

        m16_eta_rpm = arm_df[ArmDfKeys.m16.eta_rpm].values
        m25_eta_rpm = arm_df[ArmDfKeys.m25.eta_rpm].values
        m38_eta_rpm = arm_df[ArmDfKeys.m38.eta_rpm].values
        m47_eta_rpm = arm_df[ArmDfKeys.m47.eta_rpm].values

        ax1.grid(True)
        ax1.set_ylabel(r"$P_1$ + $P_6$, W")
        ax1.scatter(x=m16_eta_rpm, y=m16_pow_esc, s=2, alpha=0.5)
        ax1.axes.xaxis.set_ticklabels([])

        ax2.grid(True)
        ax2.set_ylabel(r"$P_2$ + $P_5$, W")
        ax2.scatter(x=m25_eta_rpm, y=m25_pow_esc, s=2, alpha=0.5)
        ax2.axes.xaxis.set_ticklabels([])

        ax3.grid(True)
        ax3.set_ylabel(r"$P_3$ + $P_8$, W")
        ax3.scatter(x=m38_eta_rpm, y=m38_pow_esc,
                    s=2, alpha=0.5, label='esc')
        # ax3.scatter(x=m38_eta_rpm, y=m38_pow_ars,
        #             s=2, alpha=0.5, label='ars')
        # ax3.legend(ncol=4, loc='lower left')
        ax3.axes.xaxis.set_ticklabels([])

        ax4.grid(True)
        ax4.set_ylabel(r"$P_4$ + $P_7$, W")
        ax4.scatter(x=m47_eta_rpm, y=m47_pow_esc,
                    s=2, alpha=0.5, label='esc')
        # ax4.scatter(x=m47_eta_rpm, y=m47_pow_ars,
        #             s=2, alpha=0.5, label='ars')
        # ax4.legend(ncol=4, loc='lower left')
        ax4.set_xlabel(r"Speed ratio $\eta_{\Omega}$")
        # ax4.set_xlabel("eta_rpm")

        xmin = 0.2
        xmax = 1.6
        ymin = 200
        ymax = 500
        FUPlotPower.set_axes_limits(
            [ax1, ax2, ax3, ax4],
            [[xmin, xmax], [xmin, xmax], [xmin, xmax], [xmin, xmax]],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]])

        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

    def pow_vs_time(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        # fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.figsize)
        # fig.suptitle(f'{file_tag}')
        mfs = (self.figsize[0]*0.83, self.figsize[1])
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=mfs)

        m16_pow_esc = arm_df[ArmDfKeys.m16.pow_esc]
        m25_pow_esc = arm_df[ArmDfKeys.m25.pow_esc]
        m38_pow_esc = arm_df[ArmDfKeys.m38.pow_esc]
        m47_pow_esc = arm_df[ArmDfKeys.m47.pow_esc]

        # m16_pow_ars = arm_df[ArmDfKeys.m16.pow_ars']
        # m25_pow_ars = arm_df[ArmDfKeys.m25.pow_ars']
        # m38_pow_ars = arm_df[ArmDfKeys.m38.pow_ars]
        # m47_pow_ars = arm_df[ArmDfKeys.m47.pow_ars]

        m16_delta_rpm = arm_df[ArmDfKeys.m16.delta_rpm]
        m25_delta_rpm = arm_df[ArmDfKeys.m25.delta_rpm]
        m38_delta_rpm = arm_df[ArmDfKeys.m38.delta_rpm]
        m47_delta_rpm = arm_df[ArmDfKeys.m47.delta_rpm]

        [x_in, x_in_hat, nsh_cmd] = FUParser.calculate_nsh_cmd(
            firefly_df, arm_df, ulg_dict)
        _ = x_in, x_in_hat
        drpm_cmd_time = firefly_df[FireflyDfKeys.m1.rpm].index
        drpm_cmd = nsh_cmd[0, :] * 1650

        # print(nsh_dict['nsh 0.0'])

        alpha0 = 0.7

        ax0_arr = [ax1, ax2, ax3, ax4]
        fg_arr = [True, True, True, False]
        yl_arr = [r"$P_1$ + $P_6$, W", r"$P_2$ + $P_5$, W",
                   r"$P_3$ + $P_8$, W", r"$P_4$ + $P_7$, W"]
        val_arr = [m16_pow_esc, m25_pow_esc, m38_pow_esc, m47_pow_esc]
        for ax, fg, yl, val in zip(ax0_arr, fg_arr, yl_arr, val_arr):
            ax.grid(True)
            ax.set_ylabel(yl)     # , color='black'
            ax.plot(val, alpha=alpha0, color='red')      # , color='black'
            if fg:
                ax.axes.xaxis.set_ticklabels([])
            else:
                ax.set_xlabel("Time, s")

        # xval = m16_pow_esc.index
        # ax1_arr = [ax1.twinx(), ax2.twinx(), ax3.twinx(), ax4.twinx()]
        # ylb_arr = [r"$\Delta_{0}$", r"$\Delta_{0}$",
        #            r"$\Delta_{0}$", r"$\Delta_{0}$"]
        # val_arr = [nsh_cmd[0, :], nsh_cmd[1, :], nsh_cmd[2, :], nsh_cmd[3, :]]
        # for ax, ylbl, val in zip(ax1_arr, ylb_arr, val_arr):
        #     # Offset the right spine of twin2.
        #     # The ticks and label have already been
        #     # placed on the right by twinx above.
        #     ax.spines.right.set_position(("axes", 1.01))
        #     ax.yaxis.label.set_color('blue')
        #
        #     ax.grid(True)
        #     ax.set_ylabel(ylbl, color='blue')
        #     ax.plot(xval, val, color='blue', alpha=alpha0)
        #     ax.axes.xaxis.set_ticklabels([])

        xval = m16_pow_esc.index
        ax2_arr = [ax1.twinx(), ax2.twinx(), ax3.twinx(), ax4.twinx()]
        fg_arr = [True, True, True, False]
        yl_arr = [
            r"$\Omega_1$ - $\Omega_6$, rpm", r"$\Omega_2$ - $\Omega_5$, rpm",
            r"$\Omega_3$ - $\Omega_8$, rpm", r"$\Omega_4$ - $\Omega_7$, rpm"]
        val_arr = [m16_delta_rpm, m25_delta_rpm, m38_delta_rpm, m47_delta_rpm]
        for ax, fg, yl, val in zip(ax2_arr, fg_arr, yl_arr, val_arr):
            # Offset the right spine of twin2.
            # The ticks and label have already been
            # placed on the right by twinx above.
            ax.spines.right.set_position(("axes", 1.01))
            # ax.yaxis.label.set_color('red')

            ax.grid(True)
            ax.set_ylabel(yl)     # , color='red'
            ax.plot(xval, val, alpha=alpha0, label=r'actual $ \Delta \Omega $')
            ax.plot(drpm_cmd_time, drpm_cmd, alpha=alpha0,
                    label=r'cmd($\Delta_0$) $ \Delta \Omega $')
            ax.legend(ncol=2, loc='upper right', bbox_to_anchor=(1.01, 1.28),
                      framealpha=1.0)
            if fg:
                ax.axes.xaxis.set_ticklabels([])
            else:
                ax.set_xlabel("Time, s")

        ymin = 250
        ymax = 400
        FUPlotPower.set_axes_limits(
            ax0_arr,
            [[], [], [], []],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]]
        )
        ax0_arr[0].set_yticks([ymin, ymax])
        ax0_arr[1].set_yticks([ymin, ymax])
        ax0_arr[2].set_yticks([ymin, ymax])
        ax0_arr[3].set_yticks([ymin, ymax])

        # ymin = -0.5
        # ymax = +0.5
        # FUPlotPower.set_axes_limits(
        #     ax1_arr,
        #     [[], [], [], []],
        #     [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]]
        # )
        # ax1_arr[0].set_yticks([ymin, 0, ymax], colors='blue')
        # ax1_arr[1].set_yticks([ymin, 0, ymax], colors='blue')
        # ax1_arr[2].set_yticks([ymin, 0, ymax], colors='blue')
        # ax1_arr[3].set_yticks([ymin, 0, ymax], colors='blue')

        ymin = -1000
        ymax = +1000
        FUPlotPower.set_axes_limits(
            ax2_arr,
            [[], [], [], []],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]]
        )
        ax2_arr[0].set_yticks([ymin, 0, ymax])
        ax2_arr[1].set_yticks([ymin, 0, ymax])
        ax2_arr[2].set_yticks([ymin, 0, ymax])
        ax2_arr[3].set_yticks([ymin, 0, ymax])

        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

    def totpow_vs_time(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        # fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.figsize)
        # fig.suptitle(f'{file_tag}')
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        (ax1, ax2) = axes[0, :]
        (ax3, ax4) = axes[1, :]

        m16_pow_esc = arm_df[ArmDfKeys.m16.pow_esc]
        m25_pow_esc = arm_df[ArmDfKeys.m25.pow_esc]
        m38_pow_esc = arm_df[ArmDfKeys.m38.pow_esc]
        m47_pow_esc = arm_df[ArmDfKeys.m47.pow_esc]
        tot_pow_esc = m16_pow_esc + m25_pow_esc + m38_pow_esc + m47_pow_esc

        m1_rpm = firefly_df[FireflyDfKeys.m1.rpm]
        m2_rpm = firefly_df[FireflyDfKeys.m2.rpm]
        m3_rpm = firefly_df[FireflyDfKeys.m3.rpm]
        m4_rpm = firefly_df[FireflyDfKeys.m4.rpm]
        m5_rpm = firefly_df[FireflyDfKeys.m5.rpm]
        m6_rpm = firefly_df[FireflyDfKeys.m6.rpm]
        m7_rpm = firefly_df[FireflyDfKeys.m7.rpm]
        m8_rpm = firefly_df[FireflyDfKeys.m8.rpm]
        mean_tot_rpm = (m1_rpm + m2_rpm + m3_rpm + m4_rpm +
                        m5_rpm + m6_rpm + m7_rpm + m8_rpm) / 8
        net_rpm = (m1_rpm - m2_rpm + m3_rpm - m4_rpm +
                   m5_rpm - m6_rpm + m7_rpm - m8_rpm)

        [x_in, x_in_hat, nsh_cmd] = FUParser.calculate_nsh_cmd(
            firefly_df, arm_df, ulg_dict)
        _ = x_in, x_in_hat

        alpha0 = 0.7
        [hl_tpe, hl_mtr, min_mtr, max_mtr, min_dcmd, max_dcmd] = \
            FileTagData.data_for_totpow_vs_time(file_tag)

        xval = m16_pow_esc.index
        ax1_arr = [ax1]
        flag_arr = [False]
        ylb_arr = [r"Total power, W"]
        val_arr = [tot_pow_esc]
        ones_pow = tot_pow_esc / tot_pow_esc
        for ax, flag, ylbl, val in zip(ax1_arr, flag_arr, ylb_arr, val_arr):
            ax.grid(True)
            ax.set_ylabel(ylbl)
            ax.plot(val, alpha=alpha0)
            ax.plot(xval, ones_pow * hl_tpe, linestyle='--', alpha=alpha0)
            ax.axes.xaxis.set_ticklabels([])

        xval = m16_pow_esc.index
        ax2_arr = [ax2]
        # ylb_arr = [r"Commanded $\Delta_{0}$"]
        ylb_arr = [r"$\Delta_{0}$"]
        val_arr = [nsh_cmd[0, :]]
        for ax, ylbl, val in zip(ax2_arr, ylb_arr, val_arr):
            ax.grid(True)
            ax.set_ylabel(ylbl)
            ax.plot(xval, val, alpha=alpha0)
            ax.axes.xaxis.set_ticklabels([])

        xval = m16_pow_esc.index
        ax3_arr = [ax3]
        ylb_arr = [r"Mean $\Omega$, rpm"]
        val_arr = [mean_tot_rpm]
        ones_rpm = mean_tot_rpm / mean_tot_rpm
        for ax, ylbl, val in zip(ax3_arr, ylb_arr, val_arr):
            ax.grid(True)
            ax.set_ylabel(ylbl)
            ax.plot(xval, val, alpha=alpha0)
            ax.plot(xval, ones_rpm * hl_mtr, linestyle='--', alpha=alpha0)
            # ax.axes.xaxis.set_ticklabels([])
            ax.set_xlabel("Time, s")

        xval = m16_pow_esc.index
        ax4_arr = [ax4]
        ylb_arr = [r"Net spin, rpm"]
        val_arr = [net_rpm]
        # hl_net_rpm = 300
        # ones_rpm = net_rpm / net_rpm
        for ax, ylbl, val in zip(ax4_arr, ylb_arr, val_arr):
            ax.grid(True)
            ax.set_ylabel(ylbl)
            ax.plot(xval, val, alpha=alpha0)
            # ax.plot(xval, r_rate_cmd.values*(1000/0.05), alpha=alpha0)
            # ax.plot(xval, ones_rpm * hl_net_rpm, linestyle='--', alpha=alpha0)
            ax.set_xlabel("Time, s")

        ymin = 1100
        ymax = 1500
        FUPlotPower.set_axes_limits(
            ax1_arr,
            [[], [], [], []],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]]
        )
        # ax1_arr[0].set_yticks(
        #     [ymin, 1200, 1300, hl_tot_pow, 1400, ymax],
        #     [str(ymin), '1200', '', str(hl_tot_pow), '1400', str(ymax)])

        ymin = min_dcmd
        ymax = max_dcmd
        FUPlotPower.set_axes_limits(
            ax2_arr,
            [[], [], [], []],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]]
        )
        # ax2_arr[0].set_yticks([ymin, -0.25, 0, +0.25, ymax])

        ymin = min_mtr
        ymax = max_mtr
        FUPlotPower.set_axes_limits(
            ax3_arr,
            [[], [], [], []],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]]
        )
        # ax3_arr[0].set_yticks([ymin, 17, 17.5, hl_tot_rpm, 18, ymax])

        ymin = -500
        ymax = +1500
        FUPlotPower.set_axes_limits(
            ax4_arr,
            [[], [], [], []],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]]
        )
        # ax4_arr[0].set_yticks([ymin, ymax])

        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

    def algorithm(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        # fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        # (ax1, ax2) = axes[0, :]
        # (ax3, ax4) = axes[1, :]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)

        nshstats_df = FUParser.calculate_nshstats_df(
            firefly_df, arm_df, ulg_dict, file_tag)

        # mean_delta_rpm_mean = nshstats_df['mean_delta_rpm_mean'].values
        tot_pow_esc_mean = nshstats_df['tot_pow_esc_mean'].values
        m38_pow_esc_mean = nshstats_df['m38_pow_esc_mean'].values
        m47_pow_esc_mean = nshstats_df['m47_pow_esc_mean'].values
        delta_cmd = nshstats_df['delta_cmd'].values
        # rmsd_delta_rpm = nshstats_df['rmsd_delta_rpm'].values
        # print(f'delta_cmd {delta_cmd}')
        # print(f'tot_pow_esc_mean {tot_pow_esc_mean}')

        nsh_cmd = firefly_df[FireflyDfKeys.nsh_cmd]
        fcost_avg_m38 = firefly_df[FireflyDfKeys.fcost_avg_m38]
        fcost_avg_m47 = firefly_df[FireflyDfKeys.fcost_avg_m47]
        fcost_avg_tot = firefly_df[FireflyDfKeys.fcost_avg_tot]
        # fcost_avg_tot_prev = firefly_df[FireflyDfKeys.fcost_avg_tot_prev]

        alpha0 = 0.7

        ax1.grid(True)
        ax1.set_ylabel('Mean total $P$, W')
        ax1.plot(tot_pow_esc_mean / 4, label='Total / 4', marker='.')
        ax1.plot(m38_pow_esc_mean, label='$P_{3} + P_{8}$', marker='.')
        ax1.plot(m47_pow_esc_mean, label='$P_{4} + P_{7}$', marker='.')
        # ax1.plot(m38_pow_ars_mean, label='$P_{3} + P_{8}$ ars', marker='.')
        ax1.legend(ncol=4, loc='upper right', bbox_to_anchor=(1.01, 1.18),
                  framealpha=1.0)
        ax1.axes.xaxis.set_ticklabels([])
        # ax1.set_xlabel("Time, s")
        ax1t = ax1.twinx()
        ax1t.set_ylabel(r'Commanded $ \Delta_0 $')
        ax1t.plot(delta_cmd, linestyle='', marker='.', color='grey')
        ax1.axes.xaxis.set_ticklabels([])
        # ax1t.set_xlabel("Time, s")

        ####
        indx0 = 0
        # print(fcost_avg_m38)
        # print(fcost_avg_m38.iloc[indx0:])

        ax2.grid(True)
        ax2.set_ylabel('Power, W')
        ax2.plot(fcost_avg_m38.iloc[indx0:], label='$P_{3} + P_{8}$')
        ax2.plot(fcost_avg_m47.iloc[indx0:], label='$P_{4} + P_{7}$')
        ax2.plot(fcost_avg_tot.iloc[indx0:], label='Total')
        # ax2.plot(fcost_avg_tot_prev.iloc[indx0:], label='Total prev')
        # ax2.plot(m38_pow_ars_mean, label='$P_{3} + P_{8}$ ars')
        ax2.legend(ncol=4, loc='upper right', bbox_to_anchor=(1.00, 1.18),
                   framealpha=1.0)
        # ax2.axes.xaxis.set_ticklabels([])
        ax2.set_xlabel("Time, s")
        ax2t = ax2.twinx()
        ax2t.set_ylabel(r'Commanded $ \Delta_0 $')
        ax2t.plot(nsh_cmd.iloc[indx0:], color='grey')
        ax2t.set_xlabel("Time, s")

        t0 = 0      # nsh_cmd.index[indx0]
        t1 = 350    # nsh_cmd.index[-1]
        FUPlotPower.set_axes_limits(
            [ax2, ax2t],
            [[t0, t1], [t0, t1]],
            [[225, 325], [-0.8, +0.2]]
        )
        ax2.set_yticks([225, 325])
        ax2t.set_yticks([0.2, 0, -0.2, -0.4, -0.6, -0.8])
        # ax1_arr[0].set_yticks(
        #     [ymin, 1200, 1300, hl_tot_pow, 1400, ymax],
        #     [str(ymin), '1200', '', str(hl_tot_pow), '1400', str(ymax)])

        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

    def pow_vs_nshstats(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        # fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.figsize)
        # fig.suptitle(f'{file_tag}')
        fig, axes = plt.subplots(4, 2, figsize=self.figsize)

        nshstats_df = FUParser.calculate_nshstats_df(
            firefly_df, arm_df, ulg_dict, file_tag)

        m16_delta_rpm_mean = nshstats_df['m16_delta_rpm_mean'].values
        m25_delta_rpm_mean = nshstats_df['m25_delta_rpm_mean'].values
        m38_delta_rpm_mean = nshstats_df['m38_delta_rpm_mean'].values
        m47_delta_rpm_mean = nshstats_df['m47_delta_rpm_mean'].values

        # m16_eta_rpm_mean = nshstats_df['m16_eta_rpm_mean'].values
        # m25_eta_rpm_mean = nshstats_df['m25_eta_rpm_mean'].values
        # m38_eta_rpm_mean = nshstats_df['m38_eta_rpm_mean'].values
        # m47_eta_rpm_mean = nshstats_df['m47_eta_rpm_mean'].values
        delta_cmd = nshstats_df['delta_cmd'].values

        m16_pow_esc_mean = nshstats_df['m16_pow_esc_mean'].values
        m25_pow_esc_mean = nshstats_df['m25_pow_esc_mean'].values
        m38_pow_esc_mean = nshstats_df['m38_pow_esc_mean'].values
        m47_pow_esc_mean = nshstats_df['m47_pow_esc_mean'].values

        m16_pow_esc_std = nshstats_df['m16_pow_esc_std'].values
        m25_pow_esc_std = nshstats_df['m25_pow_esc_std'].values
        m38_pow_esc_std = nshstats_df['m38_pow_esc_std'].values
        m47_pow_esc_std = nshstats_df['m47_pow_esc_std'].values

        (ax1, ax2, ax3, ax4) = axes[:, 0]
        axes_left = [ax1, ax2, ax3, ax4]

        fg_arr = [True, True, True, False]
        lb_arr = ["$P_1$ + $P_6$, W", "$P_2$ + $P_5$, W",
                  "$P_3$ + $P_8$, W", "$P_4$ + $P_7$, W"]
        xr_arr = [delta_cmd, delta_cmd, delta_cmd, delta_cmd]
        yr_arr = [m16_pow_esc_mean, m25_pow_esc_mean,
                  m38_pow_esc_mean, m47_pow_esc_mean]
        zr_arr = [m16_pow_esc_std, m25_pow_esc_std,
                  m38_pow_esc_std, m47_pow_esc_std]
        for axl, fg, lb, xr, yr, zr in zip(
                axes_left, fg_arr, lb_arr, xr_arr, yr_arr, zr_arr):
            axl.grid(True)
            axl.set_ylabel(lb)
            # ax1.scatter(delta_cmd, m16_pow_esc_mean)
            axl.errorbar(xr, yr, yerr=zr,
                         linestyle='', marker='.', ecolor='red')
            if fg:
                axl.axes.xaxis.set_ticklabels([])
            else:
                axl.set_xlabel(r'Commanded $ \Delta_0 $')
                # ax4.set_xlabel(r"$\eta_{\Omega}$")

        (ax1t, ax2t, ax3t, ax4t) = axes[:, 1]
        axes_right = [ax1t, ax2t, ax3t, ax4t]

        x_expected = np.linspace(-0.6, +0.6, 10)
        y_expected = 1650 * x_expected

        # lbl_act_drpm = r'actual $ \Delta \Omega $'
        lbl_cmd_drpm = r'Ideal $ \Delta \Omega $ curve '

        flg_arr = [True, True, True, False]
        ylb_arr = [r"$\Omega_1$ - $\Omega_6$, rpm",
                   r"$\Omega_2$ - $\Omega_5$, rpm",
                   r"$\Omega_3$ - $\Omega_8$, rpm",
                   r"$\Omega_4$ - $\Omega_7$, rpm"]
        xrr_arr = [delta_cmd, delta_cmd, delta_cmd, delta_cmd]
        yrr_arr = [m16_delta_rpm_mean, m25_delta_rpm_mean,
                   m38_delta_rpm_mean, m47_delta_rpm_mean]
        for flg, axr, ylb, yrr, xrr in zip(
                flg_arr, axes_right, ylb_arr, yrr_arr, xrr_arr):

            axr.grid(True)
            axr.set_ylabel(ylb, labelpad=-3)
            axr.plot(x_expected, y_expected,
                     label=lbl_cmd_drpm, linestyle='--', color='black')
            axr.scatter(xrr, yrr, s=10)
            # axt.legend(ncol=4, loc='upper left')
            axr.legend(ncol=4, loc='upper left', bbox_to_anchor=(-0.02, 1.08),
                       framealpha=1.0)
            # ax1t.errorbar(
            #     m16_delta_rpm_mean, m16_pow_esc_mean, yerr=m16_pow_esc_std,
            #     linestyle='', marker='.', ecolor='red')
            if flg:
                axr.axes.xaxis.set_ticklabels([])
            else:
                axr.set_xlabel(r'Commanded $ \Delta_0 $')
                # ax4t.set_xlabel(r"$\Delta \Omega$, rpm")

        ###
        [xticks_left, yticks_left, xticks_right, yticks_right] = \
            FileTagData.axes_lims_for_pow_vs_nshstats(file_tag)

        xmin = xticks_left[0]
        xmax = xticks_left[-1]
        ymin = yticks_left[0]
        ymax = yticks_left[-1]
        FUPlotPower.set_axes_limits(
            axes_left,
            [[xmin, xmax], [xmin, xmax], [xmin, xmax], [xmin, xmax]],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]]
        )
        for axl in axes_left:
            axl.set_xticks(xticks_left)
            axl.set_yticks(yticks_left)

        xmin = xticks_right[0]
        xmax = xticks_right[-1]
        ymin = yticks_right[0]
        ymax = yticks_right[-1]
        FUPlotPower.set_axes_limits(
            axes_right,
            [[xmin, xmax], [xmin, xmax], [xmin, xmax], [xmin, xmax]],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]]
        )
        for axl in axes_right:
            axl.set_xticks(xticks_right)
            axl.set_yticks(yticks_right)
        ###

        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

    def totpow_vs_nshstats(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        # fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.figsize)
        # fig.suptitle(f'{file_tag}')
        fs = (self.figsize[0], self.figsize[1] / 2)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fs)

        bet_df = FUParser.load_kdecf245dp_35_2p0_0p9_bet_df()
        # print(bet_df)
        bet_pf = FileTagData.data_for_totpow_vs_nshstats(file_tag)
        # bet_eta_omega = bet_df['bet_eta_omega']
        bet_delta_rpm = bet_df['bet_delta_rpm']
        bet_coax_power = bet_df['bet_coax_power'] * 4 * bet_pf

        nshstats_df = FUParser.calculate_nshstats_df(
            firefly_df, arm_df, ulg_dict, file_tag)

        mean_delta_rpm_mean = nshstats_df['mean_delta_rpm_mean'].values
        tot_pow_esc_mean = nshstats_df['tot_pow_esc_mean'].values
        tot_pow_esc_std = nshstats_df['tot_pow_esc_std'].values
        delta_cmd = nshstats_df['delta_cmd'].values
        rmsd_delta_rpm = nshstats_df['rmsd_delta_rpm'].values

        ax1.grid(True)
        ax1.set_ylabel("Total $ P $, W")
        # ax1.scatter(delta_cmd, m16_pow_esc_mean)
        ax1.plot(bet_delta_rpm, bet_coax_power, label='BET prediction',
                 linestyle='--', color='black')
        ax1.errorbar(
            mean_delta_rpm_mean, tot_pow_esc_mean, yerr=tot_pow_esc_std,
            linestyle='', marker='.', ecolor='red')
        ax1.legend(ncol=4, loc='upper left')
        # ax1.axes.xaxis.set_ticklabels([])
        ax1.set_xlabel(r'Mean $ \Delta \Omega $ (4 arms), rpm')

        x_expected = np.linspace(-1, +1, 10)
        y_expected = 1650 * x_expected
        # lbl_act_drpm = r'actual $ \Delta \Omega $'
        lbl_cmd_drpm = r'Ideal $ \Delta \Omega $ curve'

        ax2.grid(True)
        ax2.set_ylabel(r"Mean $ \Delta \Omega $  and RMSD, rpm", labelpad=-3)
        # ax2.scatter(delta_cmd, m25_pow_esc_mean)
        ax2.errorbar(
            delta_cmd, mean_delta_rpm_mean, yerr=rmsd_delta_rpm,
            linestyle='', marker='.', ecolor='red')
        ax2.plot(x_expected, y_expected,
                 label=lbl_cmd_drpm, linestyle='--', color='black')
        ax2.legend(ncol=4, loc='upper left')
        # ax2.axes.xaxis.set_ticklabels([])
        ax2.set_xlabel(r'Commanded $ \Delta_0 $')

        [xticks_mdr, yticks_tpe, xticks_dcmd, yticks_rmsd] = \
            FileTagData.axes_lims_for_totpow_vs_nshstats(file_tag)

        FUPlotPower.set_axes_limits(
            [ax1, ax2],
            [[xticks_mdr[0], xticks_mdr[-1]], [xticks_dcmd[0], xticks_dcmd[-1]]]
            ,
            [[yticks_tpe[0], yticks_tpe[-1]], [yticks_rmsd[0], yticks_rmsd[-1]]]
        )
        ax1.set_xticks(xticks_mdr)
        ax1.set_yticks(yticks_tpe)
        ax2.set_xticks(xticks_dcmd)
        ax2.set_yticks(yticks_rmsd)

        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

    def pow_vs_rpm(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        fig, axes = plt.subplots(4, 2, figsize=self.figsize)

        bet_df = FUParser.load_kdecf245dp_35_2p0_0p9_bet_df()
        # print(bet_df)
        bet_pf = FileTagData.data_for_totpow_vs_nshstats(file_tag)
        bet_eta_omega = bet_df['bet_eta_omega']
        bet_delta_rpm = bet_df['bet_delta_rpm']
        bet_coax_power = bet_df['bet_coax_power'] * bet_pf

        m16_eta_rpm = arm_df[ArmDfKeys.m16.eta_rpm].values
        m25_eta_rpm = arm_df[ArmDfKeys.m25.eta_rpm].values
        m38_eta_rpm = arm_df[ArmDfKeys.m38.eta_rpm].values
        m47_eta_rpm = arm_df[ArmDfKeys.m47.eta_rpm].values

        m16_delta_rpm = arm_df[ArmDfKeys.m16.delta_rpm].values
        m25_delta_rpm = arm_df[ArmDfKeys.m25.delta_rpm].values
        m38_delta_rpm = arm_df[ArmDfKeys.m38.delta_rpm].values
        m47_delta_rpm = arm_df[ArmDfKeys.m47.delta_rpm].values

        m16_pow_esc = arm_df[ArmDfKeys.m16.pow_esc].values
        m25_pow_esc = arm_df[ArmDfKeys.m25.pow_esc].values
        m38_pow_esc = arm_df[ArmDfKeys.m38.pow_esc].values
        m47_pow_esc = arm_df[ArmDfKeys.m47.pow_esc].values

        powstat_dict = FUParser.calculate_powstat_dict(arm_df)
        # print(stat_dict.keys())

        # arr = stat_dict['m16_eta_rpm_mean']
        # print(f'm16_eta_rpm_mean, len {len(arr)} data {arr}')
        # arr = stat_dict['m16_eta_rpm_mean_m16_pow_esc_mean']
        # print(f'm16_eta_rpm_mean_m16_pow_esc_mean, len {len(arr)} data {arr}')

        (ax1, ax2, ax3, ax4) = axes[:, 0]
        axes_left = [ax1, ax2, ax3, ax4]

        f_arr = [True, True, True, False]

        # Plot BET predictions
        x_arr = [bet_eta_omega, bet_eta_omega, bet_eta_omega, bet_eta_omega]
        y_arr = [bet_coax_power, bet_coax_power, bet_coax_power, bet_coax_power]
        for ax, xarr, yarr, fg in zip(axes_left, x_arr, y_arr, f_arr):
            ax.plot(xarr, yarr, linestyle='--', color='black')
            if fg:
                ax.axes.xaxis.set_ticklabels([])
            else:
                pass

        l_arr = ["$P_1$ + $P_6$, W", "$P_2$ + $P_5$, W",
                 "$P_3$ + $P_8$, W", "$P_4$ + $P_7$, W"]
        x_arr = [m16_eta_rpm, m25_eta_rpm, m38_eta_rpm, m47_eta_rpm]
        y_arr = [m16_pow_esc, m25_pow_esc, m38_pow_esc, m47_pow_esc]
        for ax, lbl, xarr, yarr, fg in zip(
                axes_left, l_arr, x_arr, y_arr, f_arr):
            ax.grid(True)
            ax.set_ylabel(lbl)
            ax.scatter(x=xarr, y=yarr, s=2, alpha=0.7, color='lightgrey')
            if fg:
                ax.axes.xaxis.set_ticklabels([])
            else:
                ax.set_xlabel(r"$\eta_{\Omega}$")

        x_arr = [powstat_dict['m16_eta_rpm_mean'],
                 powstat_dict['m25_eta_rpm_mean'],
                 powstat_dict['m38_eta_rpm_mean'],
                 powstat_dict['m47_eta_rpm_mean']]
        y_arr = [powstat_dict['m16_eta_rpm_mean_m16_pow_esc_mean'],
                 powstat_dict['m25_eta_rpm_mean_m25_pow_esc_mean'],
                 powstat_dict['m38_eta_rpm_mean_m38_pow_esc_mean'],
                 powstat_dict['m47_eta_rpm_mean_m47_pow_esc_mean']]
        z_arr = [powstat_dict['m16_eta_rpm_mean_m16_pow_esc_std'],
                 powstat_dict['m25_eta_rpm_mean_m25_pow_esc_std'],
                 powstat_dict['m38_eta_rpm_mean_m38_pow_esc_std'],
                 powstat_dict['m47_eta_rpm_mean_m47_pow_esc_std']]
        for ax, xarr, yarr, zarr, fg in zip(
                axes_left, x_arr, y_arr, z_arr, f_arr):
            # ax.scatter(x=xarr, y=yarr, s=20, alpha=0.7, color='blue')
            ax.errorbar(xarr, yarr, yerr=zarr,
                        linestyle='', marker='.', ecolor='red')
            if fg:
                ax.axes.xaxis.set_ticklabels([])
            else:
                pass

        (ax1t, ax2t, ax3t, ax4t) = axes[:, 1]
        axes_right = [ax1t, ax2t, ax3t, ax4t]

        # Plot BET predictions
        x_arr = [bet_delta_rpm, bet_delta_rpm, bet_delta_rpm, bet_delta_rpm]
        y_arr = [bet_coax_power, bet_coax_power, bet_coax_power, bet_coax_power]
        for ax, xarr, yarr, fg in zip(axes_right, x_arr, y_arr, f_arr):
            ax.plot(xarr, yarr, linestyle='--', color='black')
            if fg:
                ax.axes.xaxis.set_ticklabels([])
            else:
                pass

        l_arr = ["$P_1$ + $P_6$, W", "$P_2$ + $P_5$, W",
                 "$P_3$ + $P_8$, W", "$P_4$ + $P_7$, W"]
        x_arr = [m16_delta_rpm, m25_delta_rpm, m38_delta_rpm, m47_delta_rpm]
        y_arr = [m16_pow_esc, m25_pow_esc, m38_pow_esc, m47_pow_esc]
        f_arr = [True, True, True, False]
        for ax, lbl, xarr, yarr, fg in zip(
                axes_right, l_arr, x_arr, y_arr, f_arr):
            ax.grid(True)
            # ax.set_ylabel(lbl)
            ax.scatter(x=xarr, y=yarr, s=2, alpha=0.7, color='lightgrey')
            if fg:
                ax.axes.xaxis.set_ticklabels([])
            else:
                ax.set_xlabel(r"$\Delta \Omega$, rpm")

        x_arr = [powstat_dict['m16_delta_rpm_mean'],
                 powstat_dict['m25_delta_rpm_mean'],
                 powstat_dict['m38_delta_rpm_mean'],
                 powstat_dict['m47_delta_rpm_mean']]
        y_arr = [powstat_dict['m16_delta_rpm_mean_m16_pow_esc_mean'],
                 powstat_dict['m25_delta_rpm_mean_m25_pow_esc_mean'],
                 powstat_dict['m38_delta_rpm_mean_m38_pow_esc_mean'],
                 powstat_dict['m47_delta_rpm_mean_m47_pow_esc_mean']]
        z_arr = [powstat_dict['m16_delta_rpm_mean_m16_pow_esc_std'],
                 powstat_dict['m25_delta_rpm_mean_m25_pow_esc_std'],
                 powstat_dict['m38_delta_rpm_mean_m38_pow_esc_std'],
                 powstat_dict['m47_delta_rpm_mean_m47_pow_esc_std']]
        for ax, xarr, yarr, zarr, fg in zip(
                axes_right, x_arr, y_arr, z_arr, f_arr):
            # ax.scatter(x=xarr, y=yarr, s=20, alpha=0.7, color='blue')
            ax.errorbar(xarr, yarr, yerr=zarr,
                        linestyle='', marker='.', ecolor='red')
            # print(f"xarr {[np.around(e) for e in xarr]}")
            # print(f"yarr {[np.around(e) for e in yarr]}")
            # print(f"zarr {[np.around(e) for e in zarr]}")
            if fg:
                ax.axes.xaxis.set_ticklabels([])
            else:
                pass

        ###
        [xticks_left, yticks_left, xticks_right, yticks_right] = \
            FileTagData.axes_lims_for_pow_vs_rpm(file_tag)

        xmin = xticks_left[0]
        xmax = xticks_left[-1]
        ymin = yticks_left[0]
        ymax = yticks_left[-1]
        FUPlotPower.set_axes_limits(
            axes_left,
            [[xmin, xmax], [xmin, xmax], [xmin, xmax], [xmin, xmax]],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]]
        )
        for ax in axes_left:
            ax.set_xticks(xticks_left)
            ax.set_yticks(yticks_left)

        xmin = xticks_right[0]
        xmax = xticks_right[-1]
        ymin = yticks_right[0]
        ymax = yticks_right[-1]
        FUPlotPower.set_axes_limits(
            axes_right,
            [[xmin, xmax], [xmin, xmax], [xmin, xmax], [xmin, xmax]],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]]
        )
        for ax in axes_right:
            ax.set_xticks(xticks_right)
            ax.set_yticks(yticks_right)
        ###

        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

    def pow_mean(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        # fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.figsize)
        # fig.suptitle(f'{file_tag}')
        # fig, ax = plt.subplots(4, 2, figsize=self.figsize)
        # (ax1, ax2, ax3, ax4) = ax[:, 0]

        nshstats_df = FUParser.calculate_nshstats_df(
            firefly_df, arm_df, ulg_dict, file_tag)
        # print(nshstats_df.keys())

        # Remove repeated delta_cmd with highest m16_pow_esc_std
        # iarr = [0, 1, 2, 3, 4, 11, 12, 13, 14]
        # nshstats_df.drop(index=iarr, inplace=True)
        # nshstats_df.index = nshstats_df['delta_cmd']
        # nshstats_df.drop(columns='delta_cmd', inplace=True)
        # print(nshstats_df)

        # https://plotly.com/python/parallel-coordinates-plot/
        # python -m pip install plotly==5.5.0
        # https://plotly.com/python/static-image-export/
        # python -m pip install -U kaleido
        # python -m pip install mathjax

        # key_arr = ['m16_delta_rpm_mean', 'm25_delta_rpm_mean',
        #            'm38_delta_rpm_mean', 'm47_delta_rpm_mean']
        # err_drpm = pandas.DataFrame(index=nshstats_df['delta_cmd'].index)
        # err_drpm['err_delta_rpm_mean'] = np.zeros(err_drpm.index.shape)
        # # print(f'err_drpm {err_drpm}')
        # for key in key_arr:
        #     delta_0 = nshstats_df['delta_cmd']
        #     error_rpm = nshstats_df[key] - 1650 * delta_0
        #     # print(f'delta_0 {delta_0}')
        #     # print(f'error_rpm {error_rpm}')
        #     err_drpm['err_delta_rpm_mean'] = \
        #         err_drpm['err_delta_rpm_mean'] + np.abs(error_rpm.values)
        #     # print(f'err_drpm {err_drpm}')
        # nshstats_df['err_delta_rpm_mean'] = err_drpm.values

        [upper_dict, lower_dict] = FileTagData.axes_lims_for_pow_mean(file_tag)

        upper_df = pandas.DataFrame.from_dict(
            data=upper_dict, orient='columns', dtype=None, columns=None)
        nshstats_df = nshstats_df.append(
            upper_df, verify_integrity=True, ignore_index=True)

        lower_df = pandas.DataFrame.from_dict(
            data=lower_dict, orient='columns', dtype=None, columns=None)
        nshstats_df = nshstats_df.append(
            lower_df, verify_integrity=True, ignore_index=True)

        print(nshstats_df)

        # https://plotly.com/python-api-reference/generated/
        # plotly.express.parallel_coordinates.html
        dim_arr = ['tot_pow_esc_mean',
                   # 'net_delta_rpm_mean',
                   'rmsd_delta_rpm',
                   'r_rate_cmd_mean',
                   'pqr_norm_mean',
                   # 'err_delta_rpm_mean',
                   'vel_norm_mean',
                   'delta_cmd']

        # fig = px.parallel_coordinates(
        #     nshstats_df, dimensions=dim_arr, color='delta_cmd', labels={
        #         'tot_pow_esc_mean': 'Total power, W',
        #         # 'net_delta_rpm_mean': 'Net spin, rpm',
        #         'rmsd_delta_rpm': 'RMSD (4 arms), rpm',
        #         'r_rate_cmd_mean': 'R rate cmd',
        #         'pqr_norm_mean': 'AngVel, deg/s',
        #         # 'err_delta_rpm_mean': 'Error delta speed, rpm',
        #         'vel_norm_mean': 'Vel, m/s',
        #         'delta_cmd': 'Delta cmd'},
        #     color_continuous_scale=[
        #         (0.00, "white"), (0.01, "white"),
        #         (0.01, "red"), (0.33, "red"),
        #         (0.33, "green"), (0.66, "green"),
        #         (0.66, "blue"), (0.99, "blue"),
        #         (0.99, "white"), (1.00, "white"),
        #     ],
        #     title='Mean values', width=900, height=450,
        # )
        fig = px.parallel_coordinates(
            nshstats_df, dimensions=dim_arr, color=nshstats_df.index, labels={
                'tot_pow_esc_mean': 'Total power, W',
                # 'net_delta_rpm_mean': 'Net spin, rpm',
                'rmsd_delta_rpm': 'RMSD (4 arms), rpm',
                'r_rate_cmd_mean': 'R rate cmd',
                'pqr_norm_mean': 'AngVel, deg/s',
                # 'err_delta_rpm_mean': 'Error delta speed, rpm',
                'vel_norm_mean': 'Vel, m/s',
                'delta_cmd': 'Delta cmd'},
            # color_continuous_scale=px.colors.diverging.Tealrose,
            # color_continuous_scale=px.colors.diverging.PiYG,
            # color_continuous_midpoint=16,
            color_continuous_scale=px.colors.sequential.Jet,
            # color_continuous_scale=px.colors.sequential.Agsunset,
            # color_continuous_scale=px.colors.sequential.Bluered,
            # color_continuous_scale=px.colors.sequential.Turbo,
            title='Mean values', width=900, height=450,
        )

        fig.update_layout(
            # autosize=False,
            # width=float(self.figsize[0]),
            # height=float(self.figsize[1]),
            # margin=dict(
            #     l=50,
            #     r=50,
            #     b=100,
            #     t=100,
            #     pad=4
            # ),
            # paper_bgcolor="LightSteelBlue",
            font=dict(
                family="Time New Roman",
                size=18,
                color="Black"
            )
        )

        fp = self.save_current_plot(
            file_tag, tag_arr=tag_arr, sep="_", ext='.png')
        fig.write_image(fp)

    def pow_std(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        nshstats_df = FUParser.calculate_nshstats_df(
            firefly_df, arm_df, ulg_dict, file_tag)

        [upper_dict, lower_dict] = FileTagData.axes_lims_for_pow_std(file_tag)

        upper_df = pandas.DataFrame.from_dict(
            data=upper_dict, orient='columns', dtype=None, columns=None)
        nshstats_df = nshstats_df.append(
            upper_df, verify_integrity=True, ignore_index=True)

        lower_df = pandas.DataFrame.from_dict(
            data=lower_dict, orient='columns', dtype=None, columns=None)
        nshstats_df = nshstats_df.append(
            lower_df, verify_integrity=True, ignore_index=True)

        dim_arr = ['tot_pow_esc_std',
                   # 'net_delta_rpm_std',
                   'rmsd_delta_rpm',
                   'r_rate_cmd_std',
                   'pqr_norm_std',
                   'vel_norm_std',
                   'delta_cmd']
        fig = px.parallel_coordinates(
            nshstats_df, dimensions=dim_arr, color='delta_cmd', labels={
                'tot_pow_esc_std': 'Total power, W',
                # 'net_delta_rpm_std': 'Net spin, rpm',
                'rmsd_delta_rpm': 'RMSD (4 arms), rpm',
                'r_rate_cmd_std': 'R rate cmd',
                'pqr_norm_std': 'AngVel, deg/s',
                'vel_norm_std': 'Vel, m/s',
                'delta_cmd': 'Delta cmd'},
            color_continuous_scale=[
                (0.00, "white"), (0.01, "white"),
                (0.01, "red"), (0.33, "red"),
                (0.33, "green"), (0.66, "green"),
                (0.66, "blue"), (0.99, "blue"),
                (0.99, "white"), (1.00, "white"),
                ],
            title='Std values', width=900, height=450,
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

    def pow_hist(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        fig, ax = plt.subplots(4, 2, figsize=self.figsize)

        nshstats_df = FUParser.calculate_nshstats_df(
            firefly_df, arm_df, ulg_dict, file_tag)

        # print(nshstats_df[[
        #     'nsh_dict_keys', 'time0', 'time1', 'net_delta_rpm_mean',
        #     'tot_pow_esc_mean', 'tot_pow_esc_std', 'delta_cmd']])
        # mlist = ['delta_cmd', 'cmd_delta_rpm',
        #          'mean_delta_rpm_mean', 'error_delta_rpm', \
        #          'net_delta_rpm_mean',
        #          'tot_pow_esc_mean', 'tot_pow_esc_std']
        rn_dict = {
            'delta_cmd': 'delta_cmd',
            'cmd_delta_rpm': 'cmd_drpm',
            # 'm16_delta_rpm_mean': 'm16_drpm_mean',
            # 'm25_delta_rpm_mean': 'm25_drpm_mean',
            # 'm38_delta_rpm_mean': 'm38_drpm_mean',
            # 'm47_delta_rpm_mean': 'm47_drpm_mean',
            'mean_delta_rpm_mean': 'mean_drpm_mean',
            'rmsd_delta_rpm': 'rmsd_drpm',
            'net_delta_rpm_mean': 'net_drpm_mean',
            'tot_pow_esc_mean': 'tpow_mean',
            'tot_pow_esc_std': 'tpow_std',
            # 'm38_pow_ars_mean': 'm38_pow_ars',
            'nsh_dict_keys': 'nsh_dict_keys',
        }
        df = nshstats_df.rename(columns=rn_dict)
        mlist = [rn_dict[k] for k in rn_dict.keys()]
        print(df[mlist])

        [nsh_key_ref, cmd_ref, ref_tpe_mean, nsh_dict_keys] = \
            FileTagData.data_for_pow_hist(file_tag)
        FUParser.print_latex_table_total_power(nshstats_df, ref_tpe_mean)

        nsh_dict = FUParser.calculate_nsh_dict(
            firefly_df, arm_df, ulg_dict, file_tag, False)
        # print(f'[pow_hist] nsh_dict.keys() {nsh_dict.keys()}')
        nsh_data_ref = nsh_dict[nsh_key_ref]

        (ax1, ax2, ax3, ax4) = ax[:, 0]
        (ax1t, ax2t, ax3t, ax4t) = ax[:, 1]
        ax_arr = [ax1, ax2, ax3, ax4, ax1t, ax2t, ax3t, ax4t]
        flag_arr = [True, True, True, False, True, True, True, False]
        num_bins = 50

        for nsh_key, axt, flag in zip(nsh_dict_keys, ax_arr, flag_arr):
            if (nsh_key == 'nsh_arr') or (nsh_key == 'nsh_diff'):
                continue
            else:
                nsh_data = nsh_dict[nsh_key]
                # print(f'nsh_data.keys() {nsh_data.keys()}')

            split_arr = nsh_key.split('_')
            # nsh_key = f'i0_{i0}_i1_{i1}_nsh_{nsh_cmd}'
            # split_arr = 0   1   2   3   4   5
            # i0 = int(split_arr[1])
            # i1 = int(split_arr[3])
            cmd = float(split_arr[5])
            # print(f'i0 {i0}, i1 {i1}, cmd {cmd}')

            # axt.set_ylabel("$| f(x) |$")
            axt.set_ylabel("$ pdf $")

            data_ref = nsh_data_ref['tot_pow_esc']
            mu_ref, sigma_ref = stats.norm.fit(data_ref)
            lbl_ref = f'$\Delta_0$ {cmd_ref} ' \
                  f'$\mu$ {round(mu_ref)} $\sigma$ {round(sigma_ref)}'
            # data = (data - mu) / sigma
            _, bins_ref, _ = axt.hist(
                data_ref, bins=num_bins, density=True, alpha=0.5, label=lbl_ref)
            best_fit = stats.norm.pdf(bins_ref, loc=mu_ref, scale=sigma_ref)
            # best_fit = stats.norm.pdf(bins)
            axt.plot(bins_ref, best_fit, color='black')

            data = nsh_data['tot_pow_esc']
            mu, sigma = stats.norm.fit(data)
            lbl = f'$\Delta_0$ {cmd} ' \
                  f'$\mu$ {round(mu)} $\sigma$ {round(sigma)}'
            # data = (data - mu) / sigma
            _, bins, _ = axt.hist(
                data, bins=num_bins, density=True, alpha=0.5, label=lbl)
            best_fit = stats.norm.pdf(bins, loc=mu, scale=sigma)
            # best_fit = stats.norm.pdf(bins)
            axt.plot(bins, best_fit, color='black')

            axt.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.54, 1.25),
                       framealpha=1.0)
            axt.grid(True)

            per_dif = (mu - mu_ref) / mu_ref * 100
            textstr = r'$ \frac{ P(\Delta_0) - P_{ref} }{ P_{ref} }$ = '
            textstr = textstr + str(round(per_dif, 1)) + ' %'
            # these are matplotlib.patch.Patch properties
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            # place a text box in upper left in axes coords
            # axt.text(1110, 0.066, textstr, fontsize=10,
            [xmin, _, _, _] = FileTagData.axes_lims_for_pow_hist(file_tag)
            axt.text(xmin + 7, 0.04, textstr, fontsize=10,
                     verticalalignment='top', bbox=props)
            # transform=axt.transAxes,

            if flag:
                axt.axes.xaxis.set_ticklabels([])
            else:
                axt.set_xlabel('Total power, W')

        [xmin, xmax, ymin, ymax] = FileTagData.axes_lims_for_pow_hist(file_tag)
        FUPlotPower.set_axes_limits(
            [ax1, ax2, ax3, ax4],
            [[xmin, xmax], [xmin, xmax], [xmin, xmax], [xmin, xmax]],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]]
        )
        FUPlotPower.set_axes_limits(
            [ax1t, ax2t, ax3t, ax4t],
            [[xmin, xmax], [xmin, xmax], [xmin, xmax], [xmin, xmax]],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]]
        )

        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

    def pow_scatter(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.figsize)
        # fig.suptitle(f'{file_tag}: Variables likely affecting hover power')

        # ulg_angvel_df = ulg_dict[UlgDictKeys.ulg_angvel_df]
        # ulg_angvelsp_df = ulg_dict[UlgDictKeys.ulg_angvelsp_df]
        # ulg_in_df = ulg_dict[UlgDictKeys.ulg_in_df]
        # ulg_out_df = ulg_dict[UlgDictKeys.ulg_out_df]
        # ulg_pv_df = ulg_dict[UlgDictKeys.ulg_pv_df]
        # ulg_angvel_df = ulg_dict[UlgDictKeys.ulg_angvel_df]
        # ulg_accel_df = ulg_dict[UlgDictKeys.ulg_accel_df]
        # ulg_angacc_df = ulg_dict[UlgDictKeys.ulg_angacc_df]

        _ = firefly_df

        # vel_norm = ulg_pv_df[UlgPvDfKeys.vel_norm]
        # pqr_norm = ulg_angvel_df[UlgAngvelDf.pqr_norm]
        # acc_norm = ulg_accel_df[UlgAccelDf.acc_norm]
        # accel_measurment = f = T = accel - G => accel = T + G = 0 => T = -9.8
        # # acc_norm = acc_norm - np.mean(acc_norm)
        # acc_norm = acc_norm - 9.8
        # angacc_norm = ulg_angacc_df[UlgAngaccDf.angacc_norm]

        m16_pow_esc = arm_df[ArmDfKeys.m16.pow_esc]
        m25_pow_esc = arm_df[ArmDfKeys.m25.pow_esc]
        m38_pow_esc = arm_df[ArmDfKeys.m38.pow_esc]
        m47_pow_esc = arm_df[ArmDfKeys.m47.pow_esc]

        # var_norm = vel_norm
        # ax1_x = var_norm.values
        # ax2_x = var_norm.values
        # ax3_x = var_norm.values
        # ax4_x = var_norm.values
        # ax1_x_label = 'Vel, m/s'
        # ax2_x_label = 'Vel, m/s'
        # ax3_x_label = 'Vel, m/s'
        # ax4_x_label = 'Vel, m/s'

        # var_norm = pqr_norm
        # var_norm = acc_norm
        # var_norm = angacc_norm
        # var_norm = angacc_norm

        ax1_x = arm_df[ArmDfKeys.m16.rate_rpm].values
        ax2_x = arm_df[ArmDfKeys.m25.rate_rpm].values
        ax3_x = arm_df[ArmDfKeys.m38.rate_rpm].values
        ax4_x = arm_df[ArmDfKeys.m47.rate_rpm].values
        ax1_x_label = 'Rotor speed, rpm'
        ax2_x_label = 'Rotor speed, rpm'
        ax3_x_label = 'Rotor speed, rpm'
        ax4_x_label = 'Rotor speed, rpm'

        # vel_norm.plot.hist(bins=num_bins, ax=ax1)
        # ax1.grid(True)
        # ax1.set_xlabel('Vel, m/s')

        ax1.grid(True)
        ax1.set_ylabel('$P_1$ + $P_6$, W')
        ax1.scatter(ax1_x, m16_pow_esc.values)
        ax1.set_xlabel(ax1_x_label)

        ax2.grid(True)
        ax2.set_ylabel('$P_2$ + $P_5$, W')
        ax2.scatter(ax2_x, m25_pow_esc.values)
        ax2.set_xlabel(ax2_x_label)

        ax3.grid(True)
        ax3.set_ylabel('$P_3$ + $P_8$, W')
        ax3.scatter(ax3_x, m38_pow_esc.values)
        ax3.set_xlabel(ax3_x_label)

        ax4.grid(True)
        ax4.set_ylabel('$P_4$ + $P_7$, W')
        ax4.scatter(ax4_x, m47_pow_esc.values)
        ax4.set_xlabel(ax4_x_label)

        # ymin = -0.5
        # ymax = +0.5
        # FUPlotPower.set_axes_limits(
        #     [ax1, ax2, ax3, ax4],
        #     [[0, 0.3], [0, 0.3], [], []],
        #     [[0, 60], [200, 500], [0, 2], [0, 500]]
        # )

        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

    def plot_live_v4(self, firefly_df, arm_df, ulg_dict, file_tag):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.figsize)
        fig.suptitle(f'{file_tag}: Avg of last 10 s')

        ax1.grid(True)
        ax1.set_ylabel("Power (esc ars)")
        ax1.set_xlabel("Time, s")
        # ax1.plot(df[fcost])
        ax1.plot(firefly_df[FireflyDfKeys.m3.fcost] + firefly_df[
            FireflyDfKeys.m8.fcost])

        ax2.grid(True)
        ax2.set_ylabel("nsh delta")
        ax2.set_xlabel("Time, s")
        ax2.plot(firefly_df[FireflyDfKeys.nsh_cmd])

        ax3.grid(True)
        ax3.set_ylabel("Cost")
        ax3.set_xlabel("Time, s")
        ax3.plot(firefly_df[FireflyDfKeys.fcost_avg_m38], label='m38')
        ax3.plot(firefly_df[FireflyDfKeys.fcost_avg_m47], label='m38')
        ax3.legend()

        ax4.grid(True)
        ax4.set_ylabel("Cost total")
        ax4.set_xlabel("Time, s")
        ax4.plot(firefly_df[[FireflyDfKeys.fcost_avg_tot,
                             FireflyDfKeys.fcost_avg_tot_prev]])

        self.save_current_plot(
            file_tag, tag_arr=['live_v4'], sep="_", ext='.png')

    def estim_nsh_cmd(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.figsize)
        # fig.suptitle(f'{file_tag}')

        [x_in, x_in_hat, nsh_cmd] = FUParser.calculate_nsh_cmd(
            firefly_df, arm_df, ulg_dict)
        # print(f'x_in.shape {x_in.shape}')
        # print(f'x_in_hat.shape {x_in_hat.shape}')

        al = 0.5
        # ls = ':'

        m1_lbl = 'real'
        m2_lbl = 'real'
        m3_lbl = 'real'
        m4_lbl = 'real'
        m5_lbl = 'estim'
        m6_lbl = 'estim'
        m7_lbl = 'estim'
        m8_lbl = 'estim'

        ax1.grid(True)
        ax1.set_ylabel("in[0]")
        ax1.plot(x_in[0, :], label=m1_lbl, alpha=al)
        ax1.plot(x_in_hat[0, :], label=m6_lbl, alpha=al)
        # ax1.legend(loc='upper left')
        ax1.legend(ncol=2, loc='upper left')
        ax1.axes.xaxis.set_ticklabels([])

        ax2.grid(True)
        ax2.set_ylabel("in[1]")
        ax2.plot(x_in[1, :], label=m2_lbl, alpha=al)
        ax2.plot(x_in_hat[1, :], label=m5_lbl, alpha=al)
        # ax2.legend(loc='upper left')
        ax2.legend(ncol=2, loc='upper left')
        ax2.axes.xaxis.set_ticklabels([])

        ax3.grid(True)
        ax3.set_ylabel("in[2]")
        ax3.plot(x_in[2, :], label=m3_lbl, alpha=al)
        ax3.plot(x_in_hat[2, :], label=m8_lbl, alpha=al)
        # ax3.legend(loc='upper left')
        ax3.legend(ncol=4, loc='upper left')
        ax3.axes.xaxis.set_ticklabels([])

        ax4.grid(True)
        ax4.set_ylabel("in[3]")
        ax4.plot(x_in[3, :], label=m4_lbl, alpha=al)
        ax4.plot(x_in_hat[3, :], label=m7_lbl, alpha=al)
        # ax4.legend(loc='upper left')
        ax4.legend(ncol=4, loc='upper left')

        ax1t = ax1.twinx()
        ax1t.set_ylabel("nsh_cmd[0]", color='blue')
        ax1t.plot(nsh_cmd[0, :], color='blue')

        ax2t = ax2.twinx()
        ax2t.set_ylabel("nsh_cmd[1]", color='blue')
        ax2t.plot(nsh_cmd[1, :], color='blue')

        ax3t = ax3.twinx()
        ax3t.set_ylabel("nsh_cmd[2]", color='blue')
        ax3t.plot(nsh_cmd[2, :], color='blue')

        ax4t = ax4.twinx()
        ax4t.set_ylabel("nsh_cmd[3]", color='blue')
        ax4t.plot(nsh_cmd[3, :], color='blue')

        ymin = -0.05
        ymax = +0.05
        FUPlotPower.set_axes_limits(
            [ax1, ax2, ax3, ax4],
            [[], [], [], []],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [10 * ymin, 10 * ymax]]
        )

        ymin = -0.7
        ymax = +0.7
        FUPlotPower.set_axes_limits(
            [ax1t, ax2t, ax3t, ax4t],
            [[], [], [], []],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]]
        )

        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')


# class FileTag

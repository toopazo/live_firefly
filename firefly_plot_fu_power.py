import argparse
import copy
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import os
import pandas
import numpy as np

from toopazo_tools.file_folder import FileFolderTools as FFTools
from firefly_parse_fu import FUParser, UlgParserTools as UlgPT
from firefly_database import FileTagData
from firefly_parse_keys import FireflyDfKeys, UlgDictKeys
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
        self.figsize = (10, 6)

    def save_current_plot(self, file_tag, tag_arr, sep, ext):
        file_name = file_tag
        for tag in tag_arr:
            file_name = file_name + sep + str(tag)
        file_path = self.plotdir + f'/' + file_name + ext

        # plt.show()
        print(f'Saving file {file_path} ..')
        plt.savefig(file_path)
        plt.close(plt.gcf())
        # return file_path

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

    def ver1(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.figsize)
        fig.suptitle(f'{file_tag}')

        m16_pow_esc = arm_df[ArmDfKeys.m16.pow_esc]
        m25_pow_esc = arm_df[ArmDfKeys.m25.pow_esc]
        m38_pow_esc = arm_df[ArmDfKeys.m38.pow_esc]
        m47_pow_esc = arm_df[ArmDfKeys.m47.pow_esc]

        # m16_pow_ars = arm_df[ArmDfKeys.m16.pow_ars']
        # m25_pow_ars = arm_df[ArmDfKeys.m25.pow_ars']
        m38_pow_ars = arm_df[ArmDfKeys.m38.pow_ars]
        m47_pow_ars = arm_df[ArmDfKeys.m47.pow_ars]

        al = 0.5
        # ls = ':'

        ax1.grid(True)
        ax1.set_ylabel("Power, W")
        ax1.plot(m16_pow_esc, label='m1 + m6', alpha=al)
        ax1.legend(ncol=2, loc='lower left')
        ax1.axes.xaxis.set_ticklabels([])

        ax2.grid(True)
        ax2.set_ylabel("Power, W")
        ax2.plot(m25_pow_esc, label='m2 + m5', alpha=al)
        ax2.legend(ncol=2, loc='lower left')
        ax2.axes.xaxis.set_ticklabels([])

        ax3.grid(True)
        ax3.set_ylabel("Power, W")
        ax3.plot(m38_pow_esc, label='m3 + m8', alpha=al)
        ax3.plot(m38_pow_ars, label='m3 + m8 ars', alpha=al)
        ax3.legend(ncol=4, loc='lower left')
        ax3.axes.xaxis.set_ticklabels([])

        ax4.grid(True)
        ax4.set_ylabel("Power, W")
        ax4.plot(m47_pow_esc, label='m4 + m7', alpha=al)
        ax4.plot(m47_pow_ars, label='m4 + m7 ars', alpha=al)
        ax4.legend(ncol=4, loc='lower left')
        ax4.set_xlabel("Time, s")

        ymin = 200
        ymax = 500
        FUPlotPower.set_axes_limits(
            [ax1, ax2, ax3, ax4],
            [[], [], [], []],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]])

        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

    def ver2(self, firefly_df, powest_df, ulg_dict, file_tag, tag_arr):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.figsize)
        fig.suptitle(f'{file_tag}: Power estimation')

        powest_df = FUParser.get_power_estimation(
            firefly_df, powest_df, ulg_dict)

        m16_res_est1 = powest_df['m16_res_est1']
        m25_res_est1 = powest_df['m25_res_est1']
        m38_res_est1 = powest_df['m38_res_est1']
        m47_res_est1 = powest_df['m47_res_est1']

        m16_res_est2 = powest_df['m16_res_est2']
        m25_res_est2 = powest_df['m25_res_est2']
        m38_res_est2 = powest_df['m38_res_est2']
        m47_res_est2 = powest_df['m47_res_est2']

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

        ymin = -100
        ymax = +100
        FUPlotPower.set_axes_limits(
            [ax1, ax2, ax3, ax4],
            [[], [], [], []],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]]
        )

        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

    def ver3(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.figsize)
        fig.suptitle(f'{file_tag}')

        # m16_pow_ars = arm_df[ArmDfKeys.m16.pow_ars'].values
        # m25_pow_ars = arm_df[ArmDfKeys.m25.pow_ars'].values
        m38_pow_ars_val = arm_df[ArmDfKeys.m38.pow_ars].values
        m47_pow_ars_val = arm_df[ArmDfKeys.m47.pow_ars].values

        m16_pow_esc_val = arm_df[ArmDfKeys.m16.pow_esc].values
        m25_pow_esc_val = arm_df[ArmDfKeys.m25.pow_esc].values
        m38_pow_esc_val = arm_df[ArmDfKeys.m38.pow_esc].values
        m47_pow_esc_val = arm_df[ArmDfKeys.m47.pow_esc].values

        m16_delta_rpm_val = arm_df[ArmDfKeys.m16.delta_rpm].values
        m25_delta_rpm_val = arm_df[ArmDfKeys.m25.delta_rpm].values
        m38_delta_rpm_val = arm_df[ArmDfKeys.m38.delta_rpm].values
        m47_delta_rpm_val = arm_df[ArmDfKeys.m47.delta_rpm].values

        # numpy.polyfit(x, y, deg)
        sorted_delta = np.sort(m16_delta_rpm_val)
        sorted_power = m16_pow_esc_val[np.argsort(np.sort(m16_delta_rpm_val))]
        print(sorted_delta)
        print(sorted_power)
        m16_polyfit = np.polyfit(
            x=sorted_delta,
            y=sorted_power,
            deg=2)
        m16_pol = np.poly1d(m16_polyfit)
        print(f'm16_pol {m16_pol}')

        ax1.grid(True)
        ax1.set_ylabel("m1 + m6, W")
        ax1.scatter(x=m16_delta_rpm_val, y=m16_pow_esc_val, s=2, alpha=0.5)
        ax1.scatter(x=m16_delta_rpm_val, y=m16_pol(m16_pow_esc_val), s=2, alpha=0.5)
        ax1.axes.xaxis.set_ticklabels([])

        ax2.grid(True)
        ax2.set_ylabel("m2 + m5, W")
        ax2.scatter(x=m25_delta_rpm_val, y=m25_pow_esc_val, s=2, alpha=0.5)
        ax2.axes.xaxis.set_ticklabels([])

        ax3.grid(True)
        ax3.set_ylabel("m3 + m8, W")
        ax3.scatter(x=m38_delta_rpm_val, y=m38_pow_esc_val,
                    s=2, alpha=0.5, label='esc')
        ax3.scatter(x=m38_delta_rpm_val, y=m38_pow_ars_val,
                    s=2, alpha=0.5, label='esc')
        ax3.legend(ncol=4, loc='lower left')
        ax3.axes.xaxis.set_ticklabels([])

        ax4.grid(True)
        ax4.set_ylabel("m4 + m7, W")
        ax4.scatter(x=m47_delta_rpm_val, y=m47_pow_esc_val,
                    s=2, alpha=0.5, label='esc')
        ax4.scatter(x=m47_delta_rpm_val, y=m47_pow_ars_val,
                    s=2, alpha=0.5, label='ars')
        ax4.legend(ncol=4, loc='lower left')
        ax4.set_xlabel("delta_rpm")

        xmin = -1500
        xmax = +1500
        ymin = 200
        ymax = 500
        FUPlotPower.set_axes_limits(
            [ax1, ax2, ax3, ax4],
            [[xmin, xmax], [xmin, xmax], [xmin, xmax], [xmin, xmax]],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]])

        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

    def ver4(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.figsize)
        fig.suptitle(f'{file_tag}')

        # m16_pow_ars = arm_df[ArmDfKeys.m16.pow_ars]
        # m25_pow_ars = arm_df[ArmDfKeys.m25.pow_ars]
        m38_pow_ars_val = arm_df[ArmDfKeys.m38.pow_ars].values
        m47_pow_ars_val = arm_df[ArmDfKeys.m47.pow_ars].values

        m16_pow_esc_val = arm_df[ArmDfKeys.m16.pow_esc].values
        m25_pow_esc_val = arm_df[ArmDfKeys.m25.pow_esc].values
        m38_pow_esc_val = arm_df[ArmDfKeys.m38.pow_esc].values
        m47_pow_esc_val = arm_df[ArmDfKeys.m47.pow_esc].values

        m16_eta_rpm_val = arm_df[ArmDfKeys.m16.eta_rpm].values
        m25_eta_rpm_val = arm_df[ArmDfKeys.m25.eta_rpm].values
        m38_eta_rpm_val = arm_df[ArmDfKeys.m38.eta_rpm].values
        m47_eta_rpm_val = arm_df[ArmDfKeys.m47.eta_rpm].values

        ax1.grid(True)
        ax1.set_ylabel("m1 + m6, W")
        ax1.scatter(x=m16_eta_rpm_val, y=m16_pow_esc_val, s=2, alpha=0.5)
        ax1.axes.xaxis.set_ticklabels([])

        ax2.grid(True)
        ax2.set_ylabel("m2 + m5, W")
        ax2.scatter(x=m25_eta_rpm_val, y=m25_pow_esc_val, s=2, alpha=0.5)
        ax2.axes.xaxis.set_ticklabels([])

        ax3.grid(True)
        ax3.set_ylabel("m3 + m8, W")
        ax3.scatter(x=m38_eta_rpm_val, y=m38_pow_esc_val,
                    s=2, alpha=0.5, label='esc')
        ax3.scatter(x=m38_eta_rpm_val, y=m38_pow_ars_val,
                    s=2, alpha=0.5, label='ars')
        ax3.legend(ncol=4, loc='lower left')
        ax3.axes.xaxis.set_ticklabels([])

        ax4.grid(True)
        ax4.set_ylabel("m4 + m7, W")
        ax4.scatter(x=m47_eta_rpm_val, y=m47_pow_esc_val,
                    s=2, alpha=0.5, label='esc')
        ax4.scatter(x=m47_eta_rpm_val, y=m47_pow_ars_val,
                    s=2, alpha=0.5, label='ars')
        ax4.legend(ncol=4, loc='lower left')
        ax4.set_xlabel("eta_rpm")

        xmin = 0.2
        xmax = 1.6
        ymin = 200
        ymax = 500
        FUPlotPower.set_axes_limits(
            [ax1, ax2, ax3, ax4],
            [[xmin, xmax], [xmin, xmax], [xmin, xmax], [xmin, xmax]],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]])

        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

    def jack(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.figsize)
        fig.suptitle(f'{file_tag}')

        #  Plot nsh_delta, delta_rpm, power vs time
        # power, eta_rpm, delta_rpm, nsh_delta vs time

        m16_pow_esc = arm_df[ArmDfKeys.m16.pow_esc]
        m25_pow_esc = arm_df[ArmDfKeys.m25.pow_esc]
        m38_pow_esc = arm_df[ArmDfKeys.m38.pow_esc]
        m47_pow_esc = arm_df[ArmDfKeys.m47.pow_esc]

        # m16_pow_ars = arm_df[ArmDfKeys.m16.pow_ars']
        # m25_pow_ars = arm_df[ArmDfKeys.m25.pow_ars']
        # m38_pow_ars = arm_df[ArmDfKeys.m38.pow_ars]
        # m47_pow_ars = arm_df[ArmDfKeys.m47.pow_ars]

        # from +-1500 to +-0.5 => 1/3000
        m16_delta_rpm_val = arm_df[ArmDfKeys.m16.delta_rpm] / 4000
        m25_delta_rpm_val = arm_df[ArmDfKeys.m25.delta_rpm] / 4000
        m38_delta_rpm_val = arm_df[ArmDfKeys.m38.delta_rpm] / 4000
        m47_delta_rpm_val = arm_df[ArmDfKeys.m47.delta_rpm] / 4000

        [x_in, x_in_hat, nsh_delta] = FUParser.calculate_nsh_delta(
            firefly_df, arm_df, ulg_dict)
        _ = x_in, x_in_hat

        al = 0.5
        # ls = ':'

        ax1.grid(True)
        ax1.set_ylabel("Power, W", color='red')
        ax1.plot(m16_pow_esc, label='m1 + m6', alpha=al, color='red')
        # ax1.legend(ncol=2, loc='lower left')
        ax1.axes.xaxis.set_ticklabels([])

        ax2.grid(True)
        ax2.set_ylabel("Power, W", color='red')
        ax2.plot(m25_pow_esc, label='m2 + m5', alpha=al, color='red')
        # ax2.legend(ncol=2, loc='lower left')
        ax2.axes.xaxis.set_ticklabels([])

        ax3.grid(True)
        ax3.set_ylabel("Power, W", color='red')
        ax3.plot(m38_pow_esc, label='m3 + m8', alpha=al, color='red')
        # ax3.plot(m38_pow_ars, label='m3 + m8 ars', alpha=al)
        # ax3.legend(ncol=4, loc='lower left')
        ax3.axes.xaxis.set_ticklabels([])

        ax4.grid(True)
        ax4.set_ylabel("Power, W", color='red')
        ax4.plot(m47_pow_esc, label='m4 + m7', alpha=al, color='red')
        # ax4.plot(m47_pow_ars, label='m4 + m7 ars', alpha=al)
        # ax4.legend(ncol=4, loc='lower left')
        ax4.set_xlabel("Time, s")

        ax1t = ax1.twinx()
        ax1t.set_ylabel("nsh_delta[0]", color='blue')
        ax1t.plot(m16_pow_esc.index, nsh_delta[0, :], color='blue', alpha=al)
        ax1t.plot(m16_delta_rpm_val, color='blue', alpha=al)

        ax2t = ax2.twinx()
        ax2t.set_ylabel("nsh_delta[1]", color='blue')
        ax2t.plot(m25_pow_esc.index, nsh_delta[1, :], color='blue', alpha=al)
        ax2t.plot(m25_delta_rpm_val, color='blue', alpha=al)

        ax3t = ax3.twinx()
        ax3t.set_ylabel("nsh_delta[2]", color='blue')
        ax3t.plot(m38_pow_esc.index, nsh_delta[2, :], color='blue', alpha=al)
        ax3t.plot(m38_delta_rpm_val, color='blue', alpha=al)

        ax4t = ax4.twinx()
        ax4t.set_ylabel("nsh_delta[3]", color='blue')
        ax4t.plot(m47_pow_esc.index, nsh_delta[3, :], color='blue', alpha=al)
        ax4t.plot(m47_delta_rpm_val, color='blue', alpha=al)

        # ymin = 200
        # ymax = 500
        # FUPlotPower.set_axes_limits(
        #     [ax1, ax2, ax3, ax4],
        #     [[], [], [], []],
        #     [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]]
        # )

        ymin = -0.5
        ymax = +0.5
        FUPlotPower.set_axes_limits(
            [ax1t, ax2t, ax3t, ax4t],
            [[], [], [], []],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]]
        )

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
        ax2.plot(firefly_df[FireflyDfKeys.nsh_delta])

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

    def nsh_delta(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.figsize)
        fig.suptitle(f'{file_tag}')

        [x_in, x_in_hat, nsh_delta] = FUParser.calculate_nsh_delta(
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
        ax1t.set_ylabel("nsh_delta[0]", color='blue')
        ax1t.plot(nsh_delta[0, :], color='blue')

        ax2t = ax2.twinx()
        ax2t.set_ylabel("nsh_delta[1]", color='blue')
        ax2t.plot(nsh_delta[1, :], color='blue')

        ax3t = ax3.twinx()
        ax3t.set_ylabel("nsh_delta[2]", color='blue')
        ax3t.plot(nsh_delta[2, :], color='blue')

        ax4t = ax4.twinx()
        ax4t.set_ylabel("nsh_delta[3]", color='blue')
        ax4t.plot(nsh_delta[3, :], color='blue')

        ymin = -0.05
        ymax = +0.05
        FUPlotPower.set_axes_limits(
            [ax1, ax2, ax3, ax4],
            [[], [], [], []],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [10 * ymin, 10 * ymax]]
        )

        ymin = -0.5
        ymax = +0.5
        FUPlotPower.set_axes_limits(
            [ax1t, ax2t, ax3t, ax4t],
            [[], [], [], []],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]]
        )

        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

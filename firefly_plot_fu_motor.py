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


class FUPlotMotor:
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

    def ver3(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.figsize)
        fig.suptitle(f'{file_tag}')

        al = 0.5
        # ls = ':'

        ff_df = firefly_df

        m1_lbl = 'esc m1'
        m2_lbl = 'esc m2'
        m3_lbl = 'esc m3'
        m4_lbl = 'esc m4'
        m5_lbl = 'esc m5'
        m6_lbl = 'esc m6'
        m7_lbl = 'esc m7'
        m8_lbl = 'esc m8'

        ax1.grid(True)
        ax1.set_ylabel("Current")
        ax1.plot(ff_df[FireflyDfKeys.m1.cur], label=m1_lbl, alpha=al)
        ax1.plot(ff_df[FireflyDfKeys.m6.cur], label=m6_lbl, alpha=al)
        # ax1.legend(loc='upper left')
        ax1.legend(ncol=2, loc='upper left')
        ax1.axes.xaxis.set_ticklabels([])

        ax2.grid(True)
        ax2.set_ylabel("Current")
        ax2.plot(ff_df[FireflyDfKeys.m2.cur], label=m2_lbl, alpha=al)
        ax2.plot(ff_df[FireflyDfKeys.m5.cur], label=m5_lbl, alpha=al)
        # ax2.legend(loc='upper left')
        ax2.legend(ncol=2, loc='upper left')
        ax2.axes.xaxis.set_ticklabels([])

        ax3.grid(True)
        ax3.set_ylabel("Current")
        ax3.plot(ff_df[FireflyDfKeys.m3.cur], label=m3_lbl, alpha=al)
        ax3.plot(ff_df[FireflyDfKeys.m8.cur], label=m8_lbl, alpha=al)
        m3_lbl = 'ars m3'
        ax3.plot(ff_df[FireflyDfKeys.m3.cur_ars], label=m3_lbl, alpha=al)
        m8_lbl = 'ars m8'
        ax3.plot(ff_df[FireflyDfKeys.m8.cur_ars], label=m8_lbl, alpha=al)
        # ax3.legend(loc='upper left')
        ax3.legend(ncol=4, loc='upper left')
        ax3.axes.xaxis.set_ticklabels([])

        ax4.grid(True)
        ax4.set_ylabel("Current")
        ax4.plot(ff_df[FireflyDfKeys.m4.cur], label=m4_lbl, alpha=al)
        ax4.plot(ff_df[FireflyDfKeys.m7.cur], label=m7_lbl, alpha=al)
        m4_lbl = 'ars m4'
        ax4.plot(ff_df[FireflyDfKeys.m4.cur_ars], label=m4_lbl, alpha=al)
        m7_lbl = 'ars m7'
        ax4.plot(ff_df[FireflyDfKeys.m7.cur_ars], label=m7_lbl, alpha=al)
        # ax4.legend(loc='upper left')
        ax4.legend(ncol=4, loc='upper left')

        ymin = 0
        ymax = 20
        FUPlotMotor.set_axes_limits(
            [ax1, ax2, ax3, ax4],
            [[], [], [], []],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]]
        )

        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

    def ver4(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.figsize)
        fig.suptitle(f'{file_tag}: Current in A')

        key_m16_delta = 'm16_delta_cur'
        key_m25_delta = 'm25_delta_cur'
        key_m38_delta = 'm38_delta_cur'
        key_m47_delta = 'm47_delta_cur'

        key_m16_eta = 'm16_eta_cur'
        key_m25_eta = 'm25_eta_cur'
        key_m38_eta = 'm38_eta_cur'
        key_m47_eta = 'm47_eta_cur'

        ax1.grid(True)
        ax1.set_ylabel('m1-m6', color='red')
        ax1.plot(arm_df[key_m16_delta], color='red')
        ax1.axes.xaxis.set_ticklabels([])

        ax1t = ax1.twinx()
        ax1t.set_ylabel("m1/m6", color='blue')
        ax1t.plot(arm_df[key_m16_eta], color='blue')

        ax2.grid(True)
        ax2.set_ylabel('m2-m5', color='red')
        ax2.plot(arm_df[key_m25_delta], color='red')
        ax2.axes.xaxis.set_ticklabels([])
        ax2t = ax2.twinx()
        ax2t.set_ylabel("m2/m5", color='blue')
        ax2t.plot(arm_df[key_m25_eta], color='blue')

        ax3.grid(True)
        ax3.set_ylabel('m3-m8', color='red')
        ax3.plot(arm_df[key_m38_delta], color='red')
        ax3.axes.xaxis.set_ticklabels([])
        ax3t = ax3.twinx()
        ax3t.set_ylabel("m3/m8", color='blue')
        ax3t.plot(arm_df[key_m38_eta], color='blue')

        ax4.grid(True)
        ax4.set_ylabel('m4-m7', color='red')
        ax4.plot(arm_df[key_m47_delta], color='red')
        ax4t = ax4.twinx()
        ax4t.set_ylabel("m4/m7", color='blue')
        ax4t.plot(arm_df[key_m47_eta], color='blue')
        ax4t.set_xlabel("Time, s")

        # ymin = -10
        # ymax = +10
        # FUPlot.set_axes_limits(
        #     [ax1, ax2, ax3, ax4],
        #     [[], [], [], []],
        #     [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]]
        # )
        # ymin = 0.2
        # ymax = 1.5
        # FUPlot.set_axes_limits(
        #     [ax1t, ax2t, ax3t, ax4t],
        #     [[], [], [], []],
        #     [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]]
        # )

        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

    def ver5(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.figsize)
        fig.suptitle(f'{file_tag}')

        ff_df = firefly_df

        prfm_dict = FUPlotMotor.kde_cf_245_dp_and_kde6213xf_185()

        ax1.grid(True)
        ax1.set_ylabel('RPM')
        ax1.scatter(x=ff_df[FireflyDfKeys.m1.cur],
                    y=ff_df[FireflyDfKeys.m1.rpm], label='m1', s=2, alpha=0.5)
        ax1.scatter(x=ff_df[FireflyDfKeys.m6.cur],
                    y=ff_df[FireflyDfKeys.m6.rpm], label='m6', s=2, alpha=0.5)
        ax1.plot(prfm_dict['cur_arr'], prfm_dict['rpm_arr'], label='kde data')
        ax1.axes.xaxis.set_ticklabels([])
        ax1.legend(loc='lower right', ncol=2)

        ax2.grid(True)
        ax2.set_ylabel('RPM')
        ax2.scatter(x=ff_df[FireflyDfKeys.m2.cur],
                    y=ff_df[FireflyDfKeys.m2.rpm], label='m2', s=2, alpha=0.5)
        ax2.scatter(x=ff_df[FireflyDfKeys.m5.cur],
                    y=ff_df[FireflyDfKeys.m5.rpm], label='m5', s=2, alpha=0.5)
        ax2.plot(prfm_dict['cur_arr'], prfm_dict['rpm_arr'], label='kde data')
        ax2.axes.xaxis.set_ticklabels([])
        ax2.legend(loc='lower right', ncol=2)

        ax3.grid(True)
        ax3.set_ylabel('RPM')
        ax3.scatter(x=ff_df[FireflyDfKeys.m3.cur],
                    y=ff_df[FireflyDfKeys.m3.rpm], label='m3', s=2, alpha=0.5)
        ax3.scatter(x=ff_df[FireflyDfKeys.m8.cur],
                    y=ff_df[FireflyDfKeys.m8.rpm], label='m8', s=2, alpha=0.5)
        ax3.scatter(x=ff_df[FireflyDfKeys.m3.cur_ars],
                    y=ff_df[FireflyDfKeys.m3.rpm], label='m3 ars', s=2, alpha=0.5)
        ax3.scatter(x=ff_df[FireflyDfKeys.m8.cur_ars],
                    y=ff_df[FireflyDfKeys.m8.rpm], label='m8 ars', s=2, alpha=0.5)
        ax3.plot(prfm_dict['cur_arr'], prfm_dict['rpm_arr'], label='kde data')
        ax3.axes.xaxis.set_ticklabels([])
        ax3.legend(loc='lower right', ncol=2)

        ax4.grid(True)
        ax4.set_ylabel('RPM')
        ax4.scatter(x=ff_df[FireflyDfKeys.m4.cur],
                    y=ff_df[FireflyDfKeys.m4.rpm], label='m4', s=2, alpha=0.5)
        ax4.scatter(x=ff_df[FireflyDfKeys.m7.cur],
                    y=ff_df[FireflyDfKeys.m7.rpm], label='m7', s=2, alpha=0.5)
        ax4.scatter(x=ff_df[FireflyDfKeys.m4.cur_ars],
                    y=ff_df[FireflyDfKeys.m4.rpm], label='m4 ars', s=2, alpha=0.5)
        ax4.scatter(x=ff_df[FireflyDfKeys.m7.cur_ars],
                    y=ff_df[FireflyDfKeys.m7.rpm], label='m7 ars', s=2, alpha=0.5)
        ax4.plot(prfm_dict['cur_arr'], prfm_dict['rpm_arr'], label='kde data')
        ax4.set_xlabel('Current')
        ax4.legend(loc='lower right', ncol=2)
        # ax4.legend(mode="expand", ncol=5, loc='lower right')

        xmin = 0
        xmax = 25
        ymin = 0
        ymax = 3000
        FUPlotMotor.set_axes_limits(
            [ax1, ax2, ax3, ax4],
            [[xmin, xmax], [xmin, xmax], [xmin, xmax], [xmin, xmax]],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]]
        )
        # ymin = 0.2
        # ymax = 1.5
        # FUPlot.set_axes_limits(
        #     [ax1t, ax2t, ax3t, ax4t],
        #     [[], [], [], []],
        #     [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]]
        # )

        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

    @staticmethod
    def kde_cf_245_dp_and_kde6213xf_185():
        # KDE-CF245-DP
        # KDE6213XF-185
        # 8S = 33.6 @ 4.3V
        performance_dict = {
            "thr_arr": np.array([0, 25, 37.5, 50, 62.5, 75, 87.5, 100]),
            "cur_arr": np.array([0, 1.8, 3.9, 7.7, 13.7, 19.7, 29.2, 38.8]),
            "pow_arr": np.array([0, 62, 135, 267, 476, 685, 1016, 1350]),
            "for_arr": np.array([0, 9.02, 16.77, 26.67, 39.32, 50.50, 64.63, 79.73]),
            "rpm_arr": np.array([0, 1680, 2280, 2820, 3480, 3900, 4380, 4800]),
        }
        return performance_dict

    def ver1(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.figsize)
        fig.suptitle(f'{file_tag}: Motor speed in rpm')

        ax1.grid(True)
        ax1.set_ylabel('m1 - m6', color='red')
        ax1.plot(arm_df[ArmDfKeys.m16.delta_rpm], color='red')
        ax1.axes.xaxis.set_ticklabels([])

        ax1t = ax1.twinx()
        ax1t.set_ylabel("m1 / m6", color='blue')
        ax1t.plot(arm_df[ArmDfKeys.m16.eta_rpm], color='blue')

        ax2.grid(True)
        ax2.set_ylabel('m2 - m5', color='red')
        ax2.plot(arm_df[ArmDfKeys.m25.delta_rpm], color='red')
        ax2.axes.xaxis.set_ticklabels([])

        ax2t = ax2.twinx()
        ax2t.set_ylabel("m2 / m5", color='blue')
        ax2t.plot(arm_df[ArmDfKeys.m25.eta_rpm], color='blue')

        ax3.grid(True)
        ax3.set_ylabel('m3 - m8', color='red')
        ax3.plot(arm_df[ArmDfKeys.m38.delta_rpm], color='red')
        ax3.axes.xaxis.set_ticklabels([])

        ax3t = ax3.twinx()
        ax3t.set_ylabel("m3 / m8", color='blue')
        ax3t.plot(arm_df[ArmDfKeys.m38.eta_rpm], color='blue')

        ax4.grid(True)
        ax4.set_ylabel('m4 - m7', color='red')
        ax4.plot(arm_df[ArmDfKeys.m47.delta_rpm], color='red')
        ax4.set_xlabel("Time, s")

        ax4t = ax4.twinx()
        ax4t.set_ylabel("m4 / m7", color='blue')
        ax4t.plot(arm_df[ArmDfKeys.m47.eta_rpm], color='blue')

        ymin = 0.5
        ymax = 1.5
        FUPlotMotor.set_axes_limits(
            [ax1t, ax2t, ax3t, ax4t],
            [[], [], [], []],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]]
        )
        ymin = -1500
        ymax = +1500
        FUPlotMotor.set_axes_limits(
            [ax1, ax2, ax3, ax4],
            [[], [], [], []],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]]
        )
        # ones_x = r_rate_cmd.index
        # ones_y = np.ones(r_rate_cmd.size)
        # ax5.plot(ones_x, ones_y * 500)

        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

    def ver2(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)
        fig.suptitle(f'{file_tag}')

        ulg_in_df = ulg_dict[UlgDictKeys.ulg_in_df]
        ulg_att_df = ulg_dict[UlgDictKeys.ulg_att_df]
        # print(arm_df.keys())

        # m16_delta_rpm - m25_delta_rpm + m38_delta_rpm - m47_delta_rpm
        # = (m1 - m6) - (m2 - m5) + (m3 - m8) - (m4 - m7)
        # = (m1 + m3 + m5 + m7) - (m2 + m4 + m6 + m8)
        # = (CCW) - (CW)
        net_delta = arm_df[ArmDfKeys.m16.delta_rpm] - arm_df[ArmDfKeys.m25.delta_rpm] + arm_df[
            ArmDfKeys.m38.delta_rpm] - arm_df[ArmDfKeys.m47.delta_rpm]

        r_rate_cmd = ulg_in_df[UlgInDfKeys.r_rate_cmd]
        sf = np.max(net_delta.values) / np.max(r_rate_cmd.values)
        r_rate_cmd = r_rate_cmd * np.abs(sf)

        yaw_angle = ulg_att_df[UlgAttDf.yaw]
        yaw_angle = yaw_angle - np.mean(yaw_angle.values)
        sf = np.max(net_delta.values) / np.max(yaw_angle.values)
        yaw_angle = yaw_angle * np.abs(sf)

        ax1.grid(True)
        ax1.set_ylabel('RPM')
        for i in range(1, 9):
            key_rpm = 'esc1%s_rpm' % i
            ax1.plot(firefly_df[key_rpm], label='m%s' % i)
        ax1.legend(mode="expand", ncol=8, loc='lower center')
        ax1.axes.xaxis.set_ticklabels([])

        ax2.grid(True)
        ax2.set_ylabel('delta, RPM')
        ax2.plot(net_delta, label='CCW - CW')
        ax2.plot(r_rate_cmd, label='R rate cmd (scaled)')
        ax2.plot(yaw_angle, label='Yaw angle (scaled)')
        ax2.legend(mode="expand", ncol=3, loc='lower center')
        # bbox_to_anchor=(0, 0.2, 1, 0.05))
        # bbox_to_anchor=(x0, y0, width, height)
        ax2.set_xlabel("Time, s")

        ymin = -2000
        # ymax = 100
        FUPlotMotor.set_axes_limits(
            [ax1, ax2],
            [[], []],
            [[0, 3000], [ymin, -ymin]]
        )

        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

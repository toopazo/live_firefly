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


class FUPlotMixer:
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

    def ca_in(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.figsize)
        fig.suptitle(f'{file_tag}')

        ulg_in_df = ulg_dict[UlgDictKeys.ulg_in_df]
        ulg_pv_df = ulg_dict[UlgDictKeys.ulg_pv_df]
        # ulg_angacc_df = ulg_dict[UlgDictKeys.ulg_angacc_df]
        # ulg_accel_df = ulg_dict[UlgDictKeys.ulg_accel_df]
        # print(ulg_accel_df.keys())
        z_pos = ulg_pv_df[UlgPvDfKeys.z]

        ax1.grid(True)
        ax1.set_ylabel('P rate cmd', color='red')
        ax1.plot(ulg_in_df[UlgInDfKeys.p_rate_cmd], color='red')

        ax2.grid(True)
        ax2.set_ylabel('Q rate cmd', color='red')
        ax2.plot(ulg_in_df[UlgInDfKeys.q_rate_cmd], color='red')

        ax3.grid(True)
        ax3.set_ylabel('R rate cmd', color='red')
        ax3.plot(ulg_in_df[UlgInDfKeys.r_rate_cmd], color='red')

        ax4.grid(True)
        ax4.set_ylabel('az cmd', color='red')
        ax4.plot(ulg_in_df[UlgInDfKeys.az_cmd], color='red')

        ax_arr = [ax1, ax2, ax3]
        FUPlotMixer.plot_add_attitude_to_twinx(ulg_dict, ax_arr)

        ax4t = ax4.twinx()
        ax4t.set_ylabel("z, m", color='blue')
        val_mean = np.mean(z_pos.values)
        lbl = 'substracted mean %s $m$' % round(val_mean, 2)
        ax4t.plot(z_pos - val_mean, color='blue', label=lbl)
        # ax4t.axes.xaxis.set_ticklabels([])
        ax4t.set_xlabel("Time, s")
        ax4t.legend(loc='lower left')

        xlim_arr = [[], [], [], []]
        ymin = -0.07
        ymax = +0.07
        ylim_arr = [[ymin, ymax], [ymin, ymax], [ymin, ymax], [0.4, 0.5]]
        FUPlotMixer.set_axes_limits([ax1, ax2, ax3, ax4], xlim_arr, ylim_arr)
        FUPlotMixer.set_axes_limits([ax4t], [[]], [[-2, +2]])

        # ones_x = ulg_in_df[UlgInDfKeys.az_cmd].index
        # ones_y = np.ones(ulg_in_df[UlgInDfKeys.az_cmd].size)
        # ax4.plot(ones_x, ones_y * az_cmd_bars[0])
        # ax4.plot(ones_x, ones_y * az_cmd_bars[1])

        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

    def ca_out(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)
        fig.suptitle(f'{file_tag}')

        ulg_out_df = ulg_dict[UlgDictKeys.ulg_out_df]

        ax1.grid(True)
        ax1.set_ylabel("Throttle, us")
        ax1.set_xlabel("Time, s")
        ax1.plot(firefly_df[FireflyDfKeys.m1.thr], label='esc11')
        ax1.plot(ulg_out_df[UlgOutDfKeys.output_0], label='output[0]')
        ax1.legend()

        ff_keys_throttle = [
            FireflyDfKeys.m1.thr, FireflyDfKeys.m2.thr,
            FireflyDfKeys.m3.thr, FireflyDfKeys.m4.thr,
            FireflyDfKeys.m5.thr, FireflyDfKeys.m6.thr,
            FireflyDfKeys.m7.thr, FireflyDfKeys.m8.thr
        ]
        ulg_throttle = [
            UlgOutDfKeys.output_0, UlgOutDfKeys.output_1,
            UlgOutDfKeys.output_2, UlgOutDfKeys.output_3,
            UlgOutDfKeys.output_4, UlgOutDfKeys.output_5,
            UlgOutDfKeys.output_6, UlgOutDfKeys.output_7,
        ]

        ax2.grid(True)
        ax2.set_ylabel("Throttle, us")
        ax2.set_xlabel("Time, s")
        ax2.plot(firefly_df[ff_keys_throttle])
        # ax2.plot(ulg_out_df[ulg_throttle])
        ulg_out_df[ulg_throttle].plot(ax=ax2, grid=True).legend(
            loc='center left')

        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

    @staticmethod
    def plot_add_attitude_to_twinx(ulg_dict, ax_arr):
        [ax1, ax2, ax3] = ax_arr
        ulg_att_df = ulg_dict[UlgDictKeys.ulg_att_df]
        roll_angle = ulg_att_df[UlgAttDf.roll]
        pitch_angle = ulg_att_df[UlgAttDf.pitch]
        yaw_angle = ulg_att_df[UlgAttDf.yaw]

        ax1t = ax1.twinx()
        ax1t.set_ylabel("Roll, deg", color='blue')
        val_mean = np.mean(roll_angle.values)
        lbl = 'substracted mean %s $deg$' % round(val_mean, 2)
        ax1t.plot(roll_angle - val_mean, color='blue', label=lbl, alpha=0.5)
        ax1t.axes.xaxis.set_ticklabels([])
        ax1t.legend(loc='lower left')
        # ax1t.legend(bbox_to_anchor=(0, 0.2, 0.4, 0.1))
        # bbox_to_anchor=(x0, y0, width, height)

        ax2t = ax2.twinx()
        ax2t.set_ylabel("Pitch, deg", color='blue')
        val_mean = np.mean(pitch_angle.values)
        lbl = 'substracted mean %s $deg$' % round(val_mean, 2)
        ax2t.plot(pitch_angle - val_mean, color='blue', label=lbl, alpha=0.5)
        ax2t.axes.xaxis.set_ticklabels([])
        ax2t.legend(loc='lower left')
        # ax2t.legend(bbox_to_anchor=(0, 0.2, 0.4, 0.1))

        ax3t = ax3.twinx()
        ax3t.set_ylabel("Yaw, deg", color='blue')
        val_mean = np.mean(yaw_angle.values)
        lbl = 'substracted mean %s $deg$' % round(val_mean, 2)
        ax3t.plot(yaw_angle - val_mean, color='blue', label=lbl, alpha=0.5)
        ax3t.axes.xaxis.set_ticklabels([])
        ax3t.legend(loc='lower left')
        # ax3t.legend(bbox_to_anchor=(0, 0.2, 0.4, 0.1))

        ymin = -4
        ymax = +4
        FUPlotMixer.set_axes_limits(
            [ax1t, ax2t, ax3t],
            [[], [], []],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax]]
        )
        return [ax1t, ax2t, ax3t]
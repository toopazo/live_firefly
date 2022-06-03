# import argparse
# import copy
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import os
import pandas
import numpy as np

# from toopazo_tools.file_folder import FileFolderTools as FFTools
# from firefly_parse_fu import FUParser, UlgParserTools as UlgPT
# from firefly_database import FileTagData
from firefly_parse_keys import FireflyDfKeys, UlgDictKeys
from firefly_parse_keys import ArmDfKeys, ArmDfKeysMi
from firefly_parse_keys import UlgInDfKeys, UlgOutDfKeys, UlgPvDfKeys, \
    UlgAngvelDf, UlgAngaccDf, UlgAttDf, UlgAccelDf


class FUPlotState:
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
        w_inches = 10
        self.figsize = (w_inches, w_inches / 2)

    def save_current_plot(self, file_tag, tag_arr, sep, ext):
        file_name = file_tag
        for tag in tag_arr:
            file_name = file_name + sep + str(tag)
        file_path = self.plotdir + f'/' + file_name + ext

        # plt.show()
        print(f'Saving file {file_path} ..')
        plt.savefig(file_path, bbox_inches='tight')
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

    def linvel(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.figsize)
        # fig.suptitle(f'{file_tag}')

        _ = firefly_df, arm_df

        # ulg_angvel_df = ulg_dict[UlgDictKeys.ulg_angvel_df]
        # ulg_angvelsp_df = ulg_dict[UlgDictKeys.ulg_angvelsp_df]
        ulg_pv_df = ulg_dict[UlgDictKeys.ulg_pv_df]

        ax1.grid(True)
        ax1.set_ylabel('Vel x, m/s')
        ax1.plot(ulg_pv_df[UlgPvDfKeys.vx])
        ax1.axes.xaxis.set_ticklabels([])

        ax2.grid(True)
        ax2.set_ylabel('Vel y, m/s')
        ax2.plot(ulg_pv_df[UlgPvDfKeys.vy])
        ax2.axes.xaxis.set_ticklabels([])

        ax3.grid(True)
        ax3.set_ylabel('Vel z, m/s')
        ax3.plot(ulg_pv_df[UlgPvDfKeys.vz])
        ax3.axes.xaxis.set_ticklabels([])

        ax4.grid(True)
        ax4.set_ylabel('Vel norm, m/s')
        ax4.plot(ulg_pv_df[UlgPvDfKeys.vel_norm])
        ax4.set_xlabel("Time, s")

        ymin = -0.5
        ymax = +0.5
        FUPlotState.set_axes_limits(
            [ax1, ax2, ax3, ax4],
            [[], [], [], []],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [0, 1]]
        )

        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

    def angvel(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.figsize)
        # fig.suptitle(f'{file_tag}')

        _ = firefly_df, arm_df

        ulg_angvel_df = ulg_dict[UlgDictKeys.ulg_angvel_df]
        # ulg_angacc_df = ulg_dict[UlgDictKeys.ulg_angacc_df]

        smooth = True
        if smooth:
            fig.suptitle('Smoothed signals (savgol_filter)')
            custom_ax1 = pandas.DataFrame(
                index=ulg_angvel_df.index, data={'data': savgol_filter(
                    ulg_angvel_df[UlgAngvelDf.roll_rate].values, 21, 3)}
            )['data']
            custom_ax2 = pandas.DataFrame(
                index=ulg_angvel_df.index, data={'data': savgol_filter(
                    ulg_angvel_df[UlgAngvelDf.pitch_rate].values, 21, 3)}
            )['data']
            custom_ax3 = pandas.DataFrame(
                index=ulg_angvel_df.index, data={'data': savgol_filter(
                    ulg_angvel_df[UlgAngvelDf.yaw_rate].values, 21, 3)}
            )['data']
            custom_ax4 = pandas.DataFrame(
                index=ulg_angvel_df.index, data={'data': savgol_filter(
                    ulg_angvel_df[UlgAngvelDf.pqr_norm].values, 21, 3)}
            )['data']
        else:
            custom_ax1 = ulg_angvel_df[UlgAngvelDf.roll_rate]
            custom_ax2 = ulg_angvel_df[UlgAngvelDf.pitch_rate]
            custom_ax3 = ulg_angvel_df[UlgAngvelDf.yaw_rate]
            custom_ax4 = ulg_angvel_df[UlgAngvelDf.pqr_norm]

        ax1.grid(True)
        ax1.set_ylabel('P, deg/s', color='red')
        ax1.plot(custom_ax1, color='red')
        ax1.axes.xaxis.set_ticklabels([])

        ax2.grid(True)
        ax2.set_ylabel('Q, deg/s', color='red')
        ax2.plot(custom_ax2, color='red')
        ax2.axes.xaxis.set_ticklabels([])

        ax3.grid(True)
        ax3.set_ylabel('R, deg/s', color='red')
        ax3.plot(custom_ax3, color='red')
        ax3.axes.xaxis.set_ticklabels([])

        ax_arr = [ax1, ax2, ax3]
        FUPlotState.plot_add_attitude_to_twinx(ulg_dict, ax_arr)

        ax4.grid(True)
        ax4.set_ylabel('PQR norm, deg/s')
        ax4.plot(custom_ax4)
        ax4.set_xlabel("Time, s")

        ymin = -5
        ymax = +5
        FUPlotState.set_axes_limits(
            [ax1, ax2, ax3, ax4],
            [[], [], [], []],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [0, 5]]
        )

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
        lbl = 'subtracted mean %s $deg$' % round(val_mean, 2)
        ax1t.plot(roll_angle - val_mean, color='blue', label=lbl, alpha=0.5)
        ax1t.axes.xaxis.set_ticklabels([])
        ax1t.legend(loc='lower left')
        # ax1t.legend(bbox_to_anchor=(0, 0.2, 0.4, 0.1))
        # bbox_to_anchor=(x0, y0, width, height)

        ax2t = ax2.twinx()
        ax2t.set_ylabel("Pitch, deg", color='blue')
        val_mean = np.mean(pitch_angle.values)
        lbl = 'subtracted mean %s $deg$' % round(val_mean, 2)
        ax2t.plot(pitch_angle - val_mean, color='blue', label=lbl, alpha=0.5)
        ax2t.axes.xaxis.set_ticklabels([])
        ax2t.legend(loc='lower left')
        # ax2t.legend(bbox_to_anchor=(0, 0.2, 0.4, 0.1))

        ax3t = ax3.twinx()
        ax3t.set_ylabel("Yaw, deg", color='blue')
        val_mean = np.mean(yaw_angle.values)
        lbl = 'subtracted mean %s $deg$' % round(val_mean, 2)
        ax3t.plot(yaw_angle - val_mean, color='blue', label=lbl, alpha=0.5)
        ax3t.axes.xaxis.set_ticklabels([])
        ax3t.legend(loc='lower left')
        # ax3t.legend(bbox_to_anchor=(0, 0.2, 0.4, 0.1))

        ymin = -4
        ymax = +4
        FUPlotState.set_axes_limits(
            [ax1t, ax2t, ax3t],
            [[], [], []],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax]]
        )
        return [ax1t, ax2t, ax3t]

    def accel(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.figsize)
        # fig.suptitle(f'{file_tag}')

        _ = firefly_df, arm_df

        ulg_accel_df = ulg_dict[UlgDictKeys.ulg_accel_df]
        # print(ulg_accel_df.keys())
        smooth = True
        if smooth:
            fig.suptitle('Smoothed signals (savgol_filter)')
            custom_ax1 = pandas.DataFrame(
                index=ulg_accel_df.index, data={'data': savgol_filter(
                    ulg_accel_df[UlgAccelDf.ax].values, 21, 3)}
            )['data']
            custom_ax2 = pandas.DataFrame(
                index=ulg_accel_df.index, data={'data': savgol_filter(
                    ulg_accel_df[UlgAccelDf.ay].values, 21, 3)}
            )['data']
            custom_ax3 = pandas.DataFrame(
                index=ulg_accel_df.index, data={'data': savgol_filter(
                    ulg_accel_df[UlgAccelDf.az].values, 21, 3)}
            )['data']
            custom_ax4 = pandas.DataFrame(
                index=ulg_accel_df.index, data={'data': savgol_filter(
                    ulg_accel_df[UlgAccelDf.acc_norm].values, 21, 3)}
            )['data']
        else:
            custom_ax1 = ulg_accel_df[UlgAccelDf.ax]
            custom_ax2 = ulg_accel_df[UlgAccelDf.ay]
            custom_ax3 = ulg_accel_df[UlgAccelDf.az]
            custom_ax4 = ulg_accel_df[UlgAccelDf.acc_norm]

        ax1.grid(True)
        ax1.set_ylabel('ax, $m/s^2$')
        ax1.plot(custom_ax1)
        ax1.axes.xaxis.set_ticklabels([])

        ax2.grid(True)
        ax2.set_ylabel('ay, $m/s^2$')
        ax2.plot(custom_ax2)
        ax2.axes.xaxis.set_ticklabels([])

        ax3.grid(True)
        ax3.set_ylabel('az, $m/s^2$')
        az_mean = np.mean(ulg_accel_df[UlgAccelDf.az].values)
        lbl = 'subtracted mean %s $m/s^2$' % round(az_mean, 2)
        ax3.plot(custom_ax3 - az_mean, label=lbl)
        ax3.axes.xaxis.set_ticklabels([])
        ax3.legend()

        ax4.grid(True)
        ax4.set_ylabel('Accel norm, $m/s^2$')
        ax4.plot(custom_ax4)
        ax4.set_xlabel("Time, s")

        ymin = -1
        ymax = +1
        FUPlotState.set_axes_limits(
            [ax1, ax2, ax3, ax4],
            [[], [], [], []],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [9, 11]]
        )

        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

    def angacc(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.figsize)
        # fig.suptitle(f'{file_tag}')

        _ = firefly_df, arm_df

        ulg_angacc_df = ulg_dict[UlgDictKeys.ulg_angacc_df]
        # ulg_att_df = ulg_dict[UlgDictKeys.ulg_att_df]
        # print(ulg_accel_df.keys())
        smooth = True
        if smooth:
            fig.suptitle('Smoothed signals (savgol_filter)')
            custom_ax1 = pandas.DataFrame(
                index=ulg_angacc_df.index, data={'data': savgol_filter(
                    ulg_angacc_df[UlgAngaccDf.p_rate].values, 21, 3)}
            )['data']
            custom_ax2 = pandas.DataFrame(
                index=ulg_angacc_df.index, data={'data': savgol_filter(
                    ulg_angacc_df[UlgAngaccDf.q_rate].values, 21, 3)}
            )['data']
            custom_ax3 = pandas.DataFrame(
                index=ulg_angacc_df.index, data={'data': savgol_filter(
                    ulg_angacc_df[UlgAngaccDf.r_rate].values, 21, 3)}
            )['data']
            custom_ax4 = pandas.DataFrame(
                index=ulg_angacc_df.index, data={'data': savgol_filter(
                    ulg_angacc_df[UlgAngaccDf.angacc_norm].values, 21, 3)}
            )['data']
        else:
            custom_ax1 = ulg_angacc_df[UlgAngaccDf.p_rate]
            custom_ax2 = ulg_angacc_df[UlgAngaccDf.q_rate]
            custom_ax3 = ulg_angacc_df[UlgAngaccDf.r_rate]
            custom_ax4 = ulg_angacc_df[UlgAngaccDf.angacc_norm]

        # roll_angle = ulg_att_df[UlgAttDf.roll]
        # pitch_angle = ulg_att_df[UlgAttDf.pitch]
        # yaw_angle = ulg_att_df[UlgAttDf.yaw]

        ax1.grid(True)
        ax1.set_ylabel('P rate, $deg/s^2$', color='red')
        ax1.plot(custom_ax1, color='red')

        ax2.grid(True)
        ax2.set_ylabel('Q rate, $deg/s^2$', color='red')
        ax2.plot(custom_ax2, color='red')

        ax3.grid(True)
        ax3.set_ylabel('R rate, $deg/s^2$', color='red')
        ax3.plot(custom_ax3, color='red')
        ax3.axes.xaxis.set_ticklabels([])

        ax_arr = [ax1, ax2, ax3]
        FUPlotState.plot_add_attitude_to_twinx(ulg_dict, ax_arr)

        ax4.grid(True)
        # ax4.set_ylabel('Angacc norm, $deg/s^2$')
        ax4.set_ylabel('Angacc norm')
        ax4.plot(custom_ax4)
        ax4.set_xlabel("Time, s")

        ymin = -200
        ymax = +200
        FUPlotState.set_axes_limits(
            [ax1, ax2, ax3, ax4],
            [[], [], [], []],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [0, 300]]
        )

        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')
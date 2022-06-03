# import argparse
# import copy
# from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import os
# import pandas
import numpy as np

# from toopazo_tools.file_folder import FileFolderTools as FFTools
# from firefly_parse_fu import FUParser, UlgParserTools as UlgPT
# from firefly_database import FileTagData
from firefly_parse_keys import FireflyDfKeys, UlgDictKeys
# from firefly_parse_keys import ArmDfKeys, ArmDfKeysMi
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

    def ca_in(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.figsize)
        # fig.suptitle(f'{file_tag}')

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
        ax4.set_xlabel("Time, s")

        ax_arr = [ax1, ax2, ax3]
        FUPlotMixer.plot_add_attitude_to_twinx(ulg_dict, ax_arr)

        ax4t = ax4.twinx()
        ax4t.set_ylabel("z, m", color='blue')
        val_mean = np.mean(z_pos.values)
        lbl = 'subtracted mean %s $m$' % round(val_mean, 2)
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
        # fig.suptitle(f'{file_tag}')

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

    def rpm_vs_thr(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.figsize)
        # fig.suptitle(f'{file_tag}: Power estimation')
        # fig.suptitle('Power estimation')

        # [x_in, x_in_hat, nsh_cmd] = FUParser.calculate_nsh_cmd(
        #     firefly_df, arm_df, ulg_dict)
        # _ = x_in, x_in_hat

        ulg_out_df = ulg_dict[UlgDictKeys.ulg_out_df]
        # thr_1 = ulg_out_df[UlgOutDfKeys.output_0]
        # thr_2 = ulg_out_df[UlgOutDfKeys.output_1]
        # thr_3 = ulg_out_df[UlgOutDfKeys.output_2]
        # thr_4 = ulg_out_df[UlgOutDfKeys.output_3]
        # thr_5 = ulg_out_df[UlgOutDfKeys.output_4]
        # thr_6 = ulg_out_df[UlgOutDfKeys.output_5]
        # thr_7 = ulg_out_df[UlgOutDfKeys.output_6]
        # thr_8 = ulg_out_df[UlgOutDfKeys.output_7]

        out_1 = (ulg_out_df[UlgOutDfKeys.output_0] - 1500) / 500
        out_2 = (ulg_out_df[UlgOutDfKeys.output_1] - 1500) / 500
        out_3 = (ulg_out_df[UlgOutDfKeys.output_2] - 1500) / 500
        out_4 = (ulg_out_df[UlgOutDfKeys.output_3] - 1500) / 500
        out_5 = (ulg_out_df[UlgOutDfKeys.output_4] - 1500) / 500
        out_6 = (ulg_out_df[UlgOutDfKeys.output_5] - 1500) / 500
        out_7 = (ulg_out_df[UlgOutDfKeys.output_6] - 1500) / 500
        out_8 = (ulg_out_df[UlgOutDfKeys.output_7] - 1500) / 500

        rpm_1 = firefly_df[FireflyDfKeys.m1.rpm]
        rpm_2 = firefly_df[FireflyDfKeys.m2.rpm]
        rpm_3 = firefly_df[FireflyDfKeys.m3.rpm]
        rpm_4 = firefly_df[FireflyDfKeys.m4.rpm]
        rpm_5 = firefly_df[FireflyDfKeys.m5.rpm]
        rpm_6 = firefly_df[FireflyDfKeys.m6.rpm]
        rpm_7 = firefly_df[FireflyDfKeys.m7.rpm]
        rpm_8 = firefly_df[FireflyDfKeys.m8.rpm]

        hat_out = np.linspace(-0.49, +0.25, 100)
        hat_a = 1650
        hat_b = 2350
        hat_rpm = hat_a * hat_out + hat_b

        # ulg_in_df = ulg_dict[UlgDictKeys.ulg_in_df]
        # var1 = ulg_in_df[UlgInDfKeys.p_rate_cmd].values
        # var2 = ulg_in_df[UlgInDfKeys.q_rate_cmd].values
        # var3 = ulg_in_df[UlgInDfKeys.r_rate_cmd].values
        # var4 = ulg_in_df[UlgInDfKeys.az_cmd].values
        # var5 = nsh_cmd[0, :]
        # var6 = nsh_cmd[1, :]
        # var7 = nsh_cmd[2, :]
        # var8 = nsh_cmd[3, :]
        # ca_out_arr = []
        # ca_out_err = []
        # for prc, qrc, rrc, azcm, d01, d02, d03, d04 in zip(
        #         var1, var2, var3, var4, var5, var6, var7, var8):
        #     ca_in = [prc, qrc, rrc, azcm, d01, d02, d03, d04]
        #     x = ca_in
        #     a = 1000
        #     b = 13
        #     ca_out = a * x + b
        #     ca_out_arr.append(ca_out)

        al = 0.5
        # ls = ':'

        ax_arr = [ax1, ax2, ax3, ax4]
        fg_arr = [True, True, True, False]
        out_u_arr = [out_1, out_2, out_3, out_4]
        out_l_arr = [out_6, out_5, out_8, out_7]
        # thr_u_arr = [thr_1, thr_2, thr_3, thr_4]
        # thr_l_arr = [thr_6, thr_5, thr_8, thr_7]
        rpm_u_arr = [rpm_1, rpm_2, rpm_3, rpm_4]
        rpm_l_arr = [rpm_6, rpm_5, rpm_8, rpm_7]
        lu_arr = ['rotor 1', 'rotor 2', 'rotor 3', 'rotor 4']
        ll_arr = ['rotor 6', 'rotor 5', 'rotor 8', 'rotor 7']
        # for ax, fg, lu, ll, thr_u, thr_l, rpm_u, rpm_l in zip(
        #         ax_arr, fg_arr, lu_arr, ll_arr,
        #         thr_u_arr, thr_l_arr, rpm_u_arr, rpm_l_arr):
        for ax, fg, lu, ll, thr_u, thr_l, rpm_u, rpm_l in zip(
                ax_arr, fg_arr, lu_arr, ll_arr,
                out_u_arr, out_l_arr, rpm_u_arr, rpm_l_arr):
            ax.grid(True)
            ax.set_ylabel("$\Omega$, rpm")
            ax.plot(hat_out, hat_rpm, color='black', alpha=al, linestyle='--')
            ax.scatter(x=thr_u, y=rpm_u, s=2, alpha=al, label=lu)
            ax.scatter(x=thr_l, y=rpm_l, s=2, alpha=al, label=ll)
            ax.legend(ncol=2, loc='center right')
            if fg:
                ax.axes.xaxis.set_ticklabels([])
            else:
                # ax.set_xlabel("Throttle, $\mu s$")
                ax.set_xlabel("Output of Ctrl. Allocation")

        xmin = -0.5
        xmax = +0.5
        ymin = 1000
        ymax = 3000
        FUPlotMixer.set_axes_limits(
            [ax1, ax2, ax3, ax4],
            [[xmin, xmax], [xmin, xmax], [xmin, xmax], [xmin, xmax]],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]]
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
        FUPlotMixer.set_axes_limits(
            [ax1t, ax2t, ax3t],
            [[], [], []],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax]]
        )
        return [ax1t, ax2t, ax3t]
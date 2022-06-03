# import argparse
# import copy
# from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import os
# import pandas
import numpy as np

# from toopazo_tools.file_folder import FileFolderTools as FFTools
from firefly_parse_fu import FUParser, UlgParserTools as UlgPT
# from firefly_database import FileTagData
from firefly_parse_keys import FireflyDfKeys, FireflyDfKeysMi, UlgDictKeys
from firefly_parse_keys import ArmDfKeys, ArmDfKeysMi
# from firefly_parse_keys import UlgInDfKeys, UlgOutDfKeys, UlgPvDfKeys, \
#     UlgAngvelDf, UlgAngaccDf, UlgAttDf, UlgAccelDf


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

    def cur_vs_time(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        fig, axes = plt.subplots(4, 2, figsize=self.figsize)

        (ax1, ax2, ax3, ax4) = axes[:, 0]
        ax_arr = [ax1, ax2, ax3, ax4]
        flag_arr = [True, True, True, False]
        mi_arr = [(1, 6), (2, 5), (3, 8), (4, 7)]
        for mi, ax, flag in zip(mi_arr, ax_arr, flag_arr):
            ax.grid(True)
            ax.set_ylabel(r'$I$, A')
            mi_u = mi[0]
            mi_l = mi[1]
            ax.plot(firefly_df[FireflyDfKeysMi(mi_u).cur], label='upper')
            ax.plot(firefly_df[FireflyDfKeysMi(mi_l).cur], label='lower')
            # ax.legend(mode="expand", ncol=8, loc='lower center')
            ax.legend(ncol=2, loc='upper center')
            if flag:
                ax.axes.xaxis.set_ticklabels([])
            else:
                ax.set_xlabel('Time, s')

        (ax1t, ax2t, ax3t, ax4t) = axes[:, 1]
        axt_arr = [ax1t, ax2t, ax3t, ax4t]
        flag_arr = [True, True, True, False, True, True, True, False]
        # m16_delta_rpm - m25_delta_rpm + m38_delta_rpm - m47_delta_rpm
        # = (m1 - m6) - (m2 - m5) + (m3 - m8) - (m4 - m7)
        # = (m1 + m3 + m5 + m7) - (m2 + m4 + m6 + m8)
        # = (CCW) - (CW)
        mi_arr = [16, 25, 38, 47]
        for mi, axt, flag in zip(mi_arr, axt_arr, flag_arr):
            axt.grid(True)
            axt.set_ylabel(r'$\Delta \ I$, A', labelpad=-5)
            axt.plot(arm_df[ArmDfKeysMi(mi).delta_cur])
            if flag:
                axt.axes.xaxis.set_ticklabels([])
            else:
                axt.set_xlabel('Time, s')

        xmin = None
        xmax = None
        ymin = 0
        ymax = 20
        FUPlotMotor.set_axes_limits(
            ax_arr,
            [[xmin, xmax], [xmin, xmax], [xmin, xmax], [xmin, xmax]],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]]
        )
        ax_arr[0].set_yticks([ymin, 10, ymax])
        ax_arr[1].set_yticks([ymin, 10, ymax])
        ax_arr[2].set_yticks([ymin, 10, ymax])
        ax_arr[3].set_yticks([ymin, 10, ymax])

        xmin = None
        xmax = None
        ymin = -10
        ymax = +10
        FUPlotMotor.set_axes_limits(
            axt_arr,
            [[xmin, xmax], [xmin, xmax], [xmin, xmax], [xmin, xmax]],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]]
        )

        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

    def delta_eta_cur(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.figsize)
        # fig.suptitle(f'{file_tag}: Current in A')

        key_m16_delta = 'm16_delta_cur'
        key_m25_delta = 'm25_delta_cur'
        key_m38_delta = 'm38_delta_cur'
        key_m47_delta = 'm47_delta_cur'

        key_m16_eta = 'm16_eta_cur'
        key_m25_eta = 'm25_eta_cur'
        key_m38_eta = 'm38_eta_cur'
        key_m47_eta = 'm47_eta_cur'

        ax1.grid(True)
        ax1.set_ylabel(r'$I_1$ - $I_6$, A', color='red')
        ax1.plot(arm_df[key_m16_delta], color='red')
        ax1.axes.xaxis.set_ticklabels([])

        ax1t = ax1.twinx()
        ax1t.set_ylabel(r"$I_1$ / $I_6$", color='blue')
        ax1t.plot(arm_df[key_m16_eta], color='blue')

        ax2.grid(True)
        ax2.set_ylabel(r'$I_2$ - $I_5$, A', color='red')
        ax2.plot(arm_df[key_m25_delta], color='red')
        ax2.axes.xaxis.set_ticklabels([])
        ax2t = ax2.twinx()
        ax2t.set_ylabel(r"$I_2$ / $I_5$", color='blue')
        ax2t.plot(arm_df[key_m25_eta], color='blue')

        ax3.grid(True)
        ax3.set_ylabel(r'$I_3$ - $I_8$, A', color='red')
        ax3.plot(arm_df[key_m38_delta], color='red')
        ax3.axes.xaxis.set_ticklabels([])
        ax3t = ax3.twinx()
        ax3t.set_ylabel(r"$I_3$ / $I_8$", color='blue')
        ax3t.plot(arm_df[key_m38_eta], color='blue')

        ax4.grid(True)
        ax4.set_ylabel(r'$I_4$ - $I_7$, A', color='red')
        ax4.plot(arm_df[key_m47_delta], color='red')
        ax4t = ax4.twinx()
        ax4t.set_ylabel(r"$I_4$ / $I_7$", color='blue')
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

    def cur_calibration(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.figsize)
        # fig.suptitle(f'{file_tag}')

        ff_df = firefly_df

        prfm_dict = FUPlotMotor.kde_cf_245_dp_and_kde6213xf_185()

        ax1.grid(True)
        ax1.set_ylabel(r'$\Omega$, rpm')
        ax1.scatter(x=ff_df[FireflyDfKeys.m1.cur],
                    y=ff_df[FireflyDfKeys.m1.rpm], label='m1', s=2, alpha=0.5)
        ax1.scatter(x=ff_df[FireflyDfKeys.m6.cur],
                    y=ff_df[FireflyDfKeys.m6.rpm], label='m6', s=2, alpha=0.5)
        ax1.plot(prfm_dict['cur_arr'], prfm_dict['rpm_arr'], label='kde data')
        ax1.axes.xaxis.set_ticklabels([])
        ax1.legend(loc='lower right', ncol=2)

        ax2.grid(True)
        ax2.set_ylabel(r'$\Omega$, rpm')
        ax2.scatter(x=ff_df[FireflyDfKeys.m2.cur],
                    y=ff_df[FireflyDfKeys.m2.rpm], label='m2', s=2, alpha=0.5)
        ax2.scatter(x=ff_df[FireflyDfKeys.m5.cur],
                    y=ff_df[FireflyDfKeys.m5.rpm], label='m5', s=2, alpha=0.5)
        ax2.plot(prfm_dict['cur_arr'], prfm_dict['rpm_arr'], label='kde data')
        ax2.axes.xaxis.set_ticklabels([])
        ax2.legend(loc='lower right', ncol=2)

        ax3.grid(True)
        ax3.set_ylabel(r'$\Omega$, rpm')
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
        ax4.set_ylabel(r'$\Omega$, rpm')
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

    def delta_eta_rpm(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.figsize)
        # fig.suptitle(f'{file_tag}: Motor speed in rpm')

        ax1.grid(True)
        ax1.set_ylabel(r'$\Omega_1$ - $\Omega_6$, rpm', color='red')
        ax1.plot(arm_df[ArmDfKeys.m16.delta_rpm], color='red')
        ax1.axes.xaxis.set_ticklabels([])

        ax1t = ax1.twinx()
        ax1t.set_ylabel(r"$\Omega_1$ / $\Omega_6$", color='blue')
        ax1t.plot(arm_df[ArmDfKeys.m16.eta_rpm], color='blue')

        ax2.grid(True)
        ax2.set_ylabel(r'$\Omega_2$ - $\Omega_5$, rpm', color='red')
        ax2.plot(arm_df[ArmDfKeys.m25.delta_rpm], color='red')
        ax2.axes.xaxis.set_ticklabels([])

        ax2t = ax2.twinx()
        ax2t.set_ylabel(r"$\Omega_2$ / $\Omega_5$", color='blue')
        ax2t.plot(arm_df[ArmDfKeys.m25.eta_rpm], color='blue')

        ax3.grid(True)
        ax3.set_ylabel(r'$\Omega_3$ - $\Omega_8$, rpm', color='red')
        ax3.plot(arm_df[ArmDfKeys.m38.delta_rpm], color='red')
        ax3.axes.xaxis.set_ticklabels([])

        ax3t = ax3.twinx()
        ax3t.set_ylabel(r"$\Omega_3$ / $\Omega_8$", color='blue')
        ax3t.plot(arm_df[ArmDfKeys.m38.eta_rpm], color='blue')

        ax4.grid(True)
        ax4.set_ylabel(r'$\Omega_4$ - $\Omega_7$, rpm', color='red')
        ax4.plot(arm_df[ArmDfKeys.m47.delta_rpm], color='red')
        ax4.set_xlabel("Time, s")

        ax4t = ax4.twinx()
        ax4t.set_ylabel(r"$\Omega_4$ / $\Omega_7$", color='blue')
        ax4t.plot(arm_df[ArmDfKeys.m47.eta_rpm], color='blue')

        ymin = 0.5
        ymax = 1.5
        FUPlotMotor.set_axes_limits(
            [ax1t, ax2t, ax3t, ax4t],
            [[], [], [], []],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]]
        )
        # ymin = -1500
        # ymax = +1500
        ymin = -300
        ymax = +300
        FUPlotMotor.set_axes_limits(
            [ax1, ax2, ax3, ax4],
            [[], [], [], []],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]]
        )
        # ones_x = r_rate_cmd.index
        # ones_y = np.ones(r_rate_cmd.size)
        # ax5.plot(ones_x, ones_y * 500)

        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

    def rpm_vs_time(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        fig, axes = plt.subplots(4, 2, figsize=self.figsize)

        [x_in, x_in_hat, nsh_cmd] = FUParser.calculate_nsh_cmd(
            firefly_df, arm_df, ulg_dict)
        _ = x_in, x_in_hat
        drpm_cmd_time = firefly_df[FireflyDfKeys.m1.rpm].index
        drpm_cmd = nsh_cmd[0, :] * 1650

        (ax1, ax2, ax3, ax4) = axes[:, 0]
        ax_arr = [ax1, ax2, ax3, ax4]
        flag_arr = [True, True, True, False]
        mi_arr = [(1, 6), (2, 5), (3, 8), (4, 7)]
        for mi, ax, flag in zip(mi_arr, ax_arr, flag_arr):
            ax.grid(True)
            ax.set_ylabel(r'$\Omega$, rpm')
            mi_u = mi[0]
            mi_l = mi[1]
            key_u = FireflyDfKeysMi(mi_u).rpm
            key_l = FireflyDfKeysMi(mi_l).rpm
            lbl_u = f'upper: mean {round(np.mean(firefly_df[key_u].values))}'
            lbl_l = f'lower: mean {round(np.mean(firefly_df[key_l].values))}'
            ax.plot(firefly_df[key_u], label=lbl_u)
            ax.plot(firefly_df[key_l], label=lbl_l)
            # ax.legend(mode="expand", ncol=8, loc='lower center')
            ax.legend(ncol=2, loc='lower center')
            if flag:
                ax.axes.xaxis.set_ticklabels([])
            else:
                ax.set_xlabel('Time, s')

        alpha0 = 0.7

        (ax1t, ax2t, ax3t, ax4t) = axes[:, 1]
        axt_arr = [ax1t, ax2t, ax3t, ax4t]
        flag_arr = [True, True, True, False, True, True, True, False]
        # m16_delta_rpm - m25_delta_rpm + m38_delta_rpm - m47_delta_rpm
        # = (m1 - m6) - (m2 - m5) + (m3 - m8) - (m4 - m7)
        # = (m1 + m3 + m5 + m7) - (m2 + m4 + m6 + m8)
        # = (CCW) - (CW)
        mi_arr = [16, 25, 38, 47]
        for mi, axt, flag in zip(mi_arr, axt_arr, flag_arr):
            axt.grid(True)
            axt.set_ylabel(r'$\Delta \ \Omega$, rpm', labelpad=-5)
            delta_rpm_key = ArmDfKeysMi(mi).delta_rpm
            axt.plot(arm_df[delta_rpm_key], alpha=alpha0,
                     label=r'actual')
            axt.plot(drpm_cmd_time, drpm_cmd, alpha=alpha0,
                     label=r'cmd($\Delta_0$)')          # $ \Delta \Omega $
            axt.legend(ncol=2, loc='upper right', bbox_to_anchor=(1.05, 1.28),
                       framealpha=1.0)
            # axt.legend(ncol=2, loc='upper right')
            if flag:
                axt.axes.xaxis.set_ticklabels([])
            else:
                axt.set_xlabel('Time, s')

        # axtt_arr = [ax1t.twinx(), ax2t.twinx(), ax3t.twinx(), ax4t.twinx()]
        # ylb_arr = [r"$\Delta_{0}$", r"$\Delta_{0}$",
        #            r"$\Delta_{0}$", r"$\Delta_{0}$"]
        # val_arr = [nsh_cmd[0, :], nsh_cmd[1, :], nsh_cmd[2, :], nsh_cmd[3, :]]
        # for ax, ylbl, val in zip(axtt_arr, ylb_arr, val_arr):
        #     # Offset the right spine of twin2.
        #     # The ticks and label have already been
        #     # placed on the right by twinx above.
        #     ax.spines.right.set_position(("axes", 1.01))
        #     ax.yaxis.label.set_color('red')
        #
        #     ax.grid(True)
        #     ax.set_ylabel(ylbl, color='red')
        #     ax.plot(nsh_cmd_time, val, color='red', alpha=alpha0)
        #     ax.axes.xaxis.set_ticklabels([])

        xmin = None
        xmax = None
        ymin = 0
        ymax = 3000
        FUPlotMotor.set_axes_limits(
            ax_arr,
            [[xmin, xmax], [xmin, xmax], [xmin, xmax], [xmin, xmax]],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]]
        )
        ax_arr[0].set_yticks([ymin, 1500, ymax])
        ax_arr[1].set_yticks([ymin, 1500, ymax])
        ax_arr[2].set_yticks([ymin, 1500, ymax])
        ax_arr[3].set_yticks([ymin, 1500, ymax])

        xmin = None
        xmax = None
        ymin = -1000
        ymax = +1000
        FUPlotMotor.set_axes_limits(
            axt_arr,
            [[xmin, xmax], [xmin, xmax], [xmin, xmax], [xmin, xmax]],
            [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]]
        )

        # xmin = None
        # xmax = None
        # ymin = -0.5
        # ymax = +0.5
        # FUPlotMotor.set_axes_limits(
        #     axtt_arr,
        #     [[xmin, xmax], [xmin, xmax], [xmin, xmax], [xmin, xmax]],
        #     [[ymin, ymax], [ymin, ymax], [ymin, ymax], [ymin, ymax]]
        # )

        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

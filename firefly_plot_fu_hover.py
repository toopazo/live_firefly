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


class FUPlotHover:
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

    def ver_norm(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.figsize)
        fig.suptitle(f'{file_tag}: Variables likely affecting hover power')

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
            fig.suptitle(f'{file_tag}: L2-norm of smoothed signals')
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

        mask = FUParser.get_hover_mask(
            norm_arr=[vel_norm, pqr_norm],
            ulg_df_arr=[ulg_pv_df, ulg_in_df, ulg_out_df],
            arm_df=arm_df)

        ax1.grid(True)
        ax1.set_ylabel('Vel, m/s')
        ax1.scatter(vel_norm[mask].index, vel_norm[mask].values*0,
                    color='red', s=10)
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

        # ymin = -0.5
        # ymax = +0.5
        FUPlotHover.set_axes_limits(
            [ax1, ax2, ax3, ax4],
            [[], [], [], []],
            [[0, 1], [0, 10], [8, 12], [0, 500]]
        )

        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

    def ver_thr(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.figsize)
        fig.suptitle(f'{file_tag}: Variables likely affecting hover power')
        _ = firefly_df, ulg_dict
        var = 'thr'
        # var = 'rpm'
        # var = 'cur'
        FUPlotHover.thr_rpm_curr(
            arm_df, plot_variable=var, ax_arr=[ax1, ax2, ax3, ax4])
        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

    def ver_rpm(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.figsize)
        fig.suptitle(f'{file_tag}: Variables likely affecting hover power')
        _ = firefly_df, ulg_dict
        # var = 'thr'
        var = 'rpm'
        # var = 'cur'
        FUPlotHover.thr_rpm_curr(
            arm_df, plot_variable=var, ax_arr=[ax1, ax2, ax3, ax4])
        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

    def ver_cur(self, firefly_df, arm_df, ulg_dict, file_tag, tag_arr):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.figsize)
        fig.suptitle(f'{file_tag}: Variables likely affecting hover power')
        _ = firefly_df, ulg_dict
        # var = 'thr'
        # var = 'rpm'
        var = 'cur'
        FUPlotHover.thr_rpm_curr(
            arm_df, plot_variable=var, ax_arr=[ax1, ax2, ax3, ax4])
        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

    @staticmethod
    def thr_rpm_curr(arm_df, plot_variable, ax_arr):
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

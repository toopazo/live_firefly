import copy
import sys

import argparse
import pandas
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import os
from live_esc.kde_uas85uvc.kdecan_plot import KdecanPlot
from live_esc.kde_uas85uvc.kdecan_parse import KdecanParser
from toopazo_ulg.parse_file import UlgParser
from toopazo_tools.file_folder import FileFolderTools as FFTools
from firefly_parse import FireflyParser


class FireflyPlot:
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
            raise RuntimeError('Directories are not present or could not be created')

        self.bdir = bdir
        self.figsize = (10, 6)

        self.firefly_parser = None
        # self.kdecan_plot = None

    def save_current_plot(self, firefly_tag, tag_arr, sep, ext):
        file_name = firefly_tag
        for tag in tag_arr:
            file_name = file_name + sep + str(tag)
        file_path = self.plotdir + f'/' + file_name + ext

        # plt.show()
        print(f'Saving file {file_path} ..')
        plt.savefig(file_path)
        # return file_path

    def process_file(self, kdecan_file, ulg_file, firefly_tag):
        self.firefly_parser = FireflyParser(self.bdir, kdecan_file, ulg_file)
        kdecan_df = self.firefly_parser.kdecan_df
        ulg_out_df = self.firefly_parser.ulg_out_df

        [kdecan_df, ulg_out_df, escid_dict] = FireflyParser.match_kdecan_and_ulg_dataframe(kdecan_df, ulg_out_df)
        self.plot_match_throttle(kdecan_df, ulg_out_df, escid_dict, firefly_tag)

        arm_dict = FireflyParser.get_kdecan_arm_dict(escid_dict)
        self.plot_arm_power(arm_dict, firefly_tag)

        # kdecan_shift = self.calculate_kdecan_shift(selected_escid=11, save_plot=True)
        # # self.plot_kdecan(kdecan_shift, firefly_tag)

    def plot_match_throttle(self, kdecan_df, ulg_out_df, escid_dict, firefly_tag):
        _ = kdecan_df
        esc11_df = escid_dict['esc11_df']
        esc12_df = escid_dict['esc12_df']
        esc13_df = escid_dict['esc13_df']
        esc14_df = escid_dict['esc14_df']
        esc15_df = escid_dict['esc15_df']
        esc16_df = escid_dict['esc16_df']
        esc17_df = escid_dict['esc17_df']
        esc18_df = escid_dict['esc18_df']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)
        ax1.grid(True)
        ax1.set_ylabel("Throttle, us")
        ax1.set_xlabel("Time, s")
        ax2.grid(True)
        ax2.set_ylabel("Throttle, us")
        ax2.set_xlabel("Time, s")

        for i in range(0, 8):
            ax1.plot(ulg_out_df[f'output[{i}]'])
            ax1.plot(esc11_df[f'inthtl us'])
            ax1.plot(esc12_df[f'inthtl us'])
            ax1.plot(esc13_df[f'inthtl us'])
            ax1.plot(esc14_df[f'inthtl us'])
            ax1.plot(esc15_df[f'inthtl us'])
            ax1.plot(esc16_df[f'inthtl us'])
            ax1.plot(esc17_df[f'inthtl us'])
            ax1.plot(esc18_df[f'inthtl us'])

        ax2.plot(ulg_out_df[f'output[0]'], label='ulg')
        ax2.plot(esc11_df[f'inthtl us'], label='kdecan')
        ax2.legend()

        self.save_current_plot(firefly_tag, tag_arr=['match_throttle'], sep="_", ext='.png')

    def plot_arm_power(self, arm_dict, firefly_tag):
        for i in range(0, 4):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)
            ax1.grid(True)
            ax1.set_ylabel("Power, W")
            ax1.set_xlabel("RPM ratio")
            ax2.grid(True)
            ax2.set_ylabel("Throttle ratio")
            ax2.set_xlabel("RPM ratio")

            arm_key = f'arm{i+1}'
            [arm_power, arm_eta_angvel, arm_eta_throttle] = arm_dict[arm_key]
            ax1.scatter(arm_eta_angvel, arm_power)
            ax2.scatter(arm_eta_angvel, arm_eta_throttle)
            self.save_current_plot(firefly_tag, tag_arr=[arm_key], sep="_", ext='.png')

    def plot_kdecan(self, kdecan_shift, firefly_tag):
        kdecan_df = self.firefly_parser.kdecan_df
        assert isinstance(kdecan_df, pandas.DataFrame)

        col_arr = [KdecanParser.col_voltage, KdecanParser.col_current, KdecanParser.col_rpm, KdecanParser.col_inthtl]
        for escid in range(11, 19):
            df_tmp = kdecan_df[kdecan_df[KdecanParser.col_escid] == escid]
            df_tmp = df_tmp[col_arr]
            df_tmp = FireflyPlot.apply_kdecan_shift(df_tmp, kdecan_shift)
            power_in = np.array(df_tmp[KdecanParser.col_voltage].values) * np.array(
                df_tmp[KdecanParser.col_current].values)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=self.figsize)
        ax1.grid(True)
        ax1.set_ylabel("correlation")
        ax1.set_xlabel("lags")
        ax2.grid(True)
        ax2.set_ylabel("throttle us")
        ax2.set_xlabel("time s")
        ax3.grid(True)
        ax3.set_ylabel("throttle us")
        ax3.set_xlabel("time s")

        # df_tmp.plot(figsize=self.figsize, subplots=True)
        # self.save_current_plot(firefly_tag, tag_arr=["escid{}".format(escid)], sep="_", ext='.png')

    def find_file(self, extension, log_num):
        selected_file = ''
        file_arr = FFTools.get_file_arr(fpath=self.logdir, extension=extension)
        for file in file_arr:
            if log_num is not None:
                pattern = f'_{log_num}_'
                if pattern in file:
                    selected_file = os.path.abspath(file)
                    break
        print(f'[find_file] logdir {self.logdir}')
        print(f'[find_file] extension {extension}')
        print(f'[find_file] log_num {log_num}')
        print(f'[find_file] selected_file {selected_file}')
        return selected_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Parse, process and plot .kdecan files')
    parser.add_argument('--bdir', action='store', required=True,
                        help='Base directory of [logs, tmp, plots] folders')
    parser.add_argument('--kdecan', action='store', required=True,
                        help='Specific log file number to process')
    parser.add_argument('--ulg', action='store', required=True,
                        help='Specific log file number to process')
    # parser.add_argument('--plot', action='store_true', required=False,
    #                     help='plot results')
    # parser.add_argument('--pos_vel', action='store_true', help='pos and vel')
    # parser.add_argument('--rpy_angles', action='store_true', help='roll, pitch and yaw attitude angles')
    # parser.add_argument('--pqr_angvel', action='store_true', help='P, Q, and R body angular velocity')
    # parser.add_argument('--man_ctrl', action='store_true', help='manual control')
    # parser.add_argument('--ctrl_alloc', action='store_true', help='Control allocation and in/out analysis')

    args = parser.parse_args()

    abs_bdir = os.path.abspath(args.bdir)
    firefly_plot = FireflyPlot(abs_bdir)
    abs_kdecan_file = firefly_plot.find_file('.kdecan', args.kdecan)
    abs_ulg_file = firefly_plot.find_file('.ulg', args.ulg)
    abs_firefly_tag = f'kdecan_log_{args.kdecan}_ulg_log_{args.ulg}'
    if os.path.isfile(abs_kdecan_file) and os.path.isfile(abs_ulg_file):
        firefly_plot.process_file(abs_kdecan_file, abs_ulg_file, abs_firefly_tag)
    else:
        raise RuntimeError(f'No such files {abs_kdecan_file} or {abs_ulg_file}')

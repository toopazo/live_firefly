import argparse
import pandas
import matplotlib.pyplot as plt
import numpy as np
import os

from toopazo_tools.file_folder import FileFolderTools as FFTools
from live_esc.kde_uas85uvc.kdecan_parse import KdecanParser
from firefly_parse_ku import KUParser
from firefly_parse_ku import EscidParserTools as EscidPT
from firefly_parse_ku import UlgParserTools as UlgPT
from firefly_parse_ku import DataframeTools


class KUPlot:
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

        self.ku_parser = None

    def save_current_plot(self, file_tag, tag_arr, sep, ext):
        file_name = file_tag
        for tag in tag_arr:
            file_name = file_name + sep + str(tag)
        file_path = self.plotdir + f'/' + file_name + ext

        # plt.show()
        print(f'Saving file {file_path} ..')
        plt.savefig(file_path)
        # return file_path

    def process_file(self, kdecan_file, ulg_file, file_tag):
        self.ku_parser = KUParser(self.bdir, kdecan_file, ulg_file)
        escid_dict = self.ku_parser.escid_dict
        ulg_dict = self.ku_parser.ulg_dict

        # Synchornize data (and reset start times)
        ulg_time_secs = DataframeTools.shortest_time_secs(ulg_dict)
        escid_time_secs = DataframeTools.shortest_time_secs(escid_dict)
        shortest_time_secs = np.linspace(
            min(ulg_time_secs[0], escid_time_secs[0]),
            max(ulg_time_secs[-1], escid_time_secs[-1]),
            min(len(ulg_time_secs), len(escid_time_secs))
        )
        max_delta = 0.01
        escid_dict = EscidPT.resample(escid_dict, shortest_time_secs, max_delta)
        ulg_dict = UlgPT.resample(ulg_dict, shortest_time_secs, max_delta)
        [escid_dict, ulg_dict] = KUParser.synchronize(escid_dict, ulg_dict)
        self.plot_match_throttle(escid_dict, ulg_dict, file_tag)

        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)
        # ref_df = escid_dict['esc11_df']
        ref_df = ulg_dict['ulg_in_df']
        ref_df.plot(subplots=True)
        self.save_current_plot(file_tag, tag_arr=['ref_df_1'], sep="_",
                               ext='.png')

        [escid_dict, ulg_dict] = KUParser.filter_by_hover(
            escid_dict, ulg_dict)
        _ = ulg_dict

        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)
        # ref_df = escid_dict['esc11_df']
        ref_df = ulg_dict['ulg_in_df']
        ref_df.plot(subplots=True)
        self.save_current_plot(file_tag, tag_arr=['ref_df_2'], sep="_",
                               ext='.png')

        arm_dict = EscidPT.get_arm_dict(escid_dict)
        self.plot_arm_power(arm_dict, file_tag)

        # kdecan_shift = self.calculate_kdecan_shift(
        #     selected_escid=11, save_plot=True)
        # # self.plot_kdecan(kdecan_shift, file_tag)

    def plot_match_throttle(self, escid_dict, ulg_dict, file_tag):
        esc11_df = escid_dict['esc11_df']
        esc12_df = escid_dict['esc12_df']
        esc13_df = escid_dict['esc13_df']
        esc14_df = escid_dict['esc14_df']
        esc15_df = escid_dict['esc15_df']
        esc16_df = escid_dict['esc16_df']
        esc17_df = escid_dict['esc17_df']
        esc18_df = escid_dict['esc18_df']

        ulg_out_df = ulg_dict['ulg_out_df']

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

        self.save_current_plot(
            file_tag, tag_arr=['match_throttle'], sep="_", ext='.png')

    def plot_arm_power(self, arm_dict, file_tag):
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
            self.save_current_plot(
                file_tag, tag_arr=[arm_key], sep="_", ext='.png')

    def plot_kdecan(self, kdecan_shift, file_tag):
        kdecan_df = self.ku_parser.kdecan_df
        assert isinstance(kdecan_df, pandas.DataFrame)

        col_arr = [KdecanParser.col_voltage, KdecanParser.col_current,
                   KdecanParser.col_rpm, KdecanParser.col_inthtl]
        for escid in range(11, 19):
            df_tmp = kdecan_df[kdecan_df[KdecanParser.col_escid] == escid]
            df_tmp = df_tmp[col_arr]
            df_tmp = KUParser.apply_kdecan_shift(df_tmp, kdecan_shift)
            # power_in = np.array(
            #     df_tmp[KdecanParser.col_voltage].values) * np.array(
            #     df_tmp[KdecanParser.col_current].values)

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

        df_tmp.plot(figsize=self.figsize, subplots=True)
        self.save_current_plot(
            file_tag, tag_arr=["escid{}".format(escid)], sep="_", ext='.png')


def find_file_in_folder(fpath, extension, log_num):
    selected_file = ''
    file_arr = FFTools.get_file_arr(fpath=fpath, extension=extension)
    for file in file_arr:
        if log_num is not None:
            pattern = f'_{log_num}_'
            if pattern in file:
                selected_file = os.path.abspath(file)
                break
    print(f'[find_file_in_folder] fpath {fpath}')
    print(f'[find_file_in_folder] extension {extension}')
    print(f'[find_file_in_folder] log_num {log_num}')
    print(f'[find_file_in_folder] selected_file {selected_file}')
    return selected_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Parse, process and plot .kdecan files')
    parser.add_argument('--bdir', action='store', required=True,
                        help='Base directory of [logs, tmp, plots] folders')
    parser.add_argument('--kdecan', action='store', required=False,
                        help='Specific log file number to process')
    parser.add_argument('--ulg', action='store', required=False,
                        help='Specific log file number to process')

    args = parser.parse_args()

    abs_bdir = os.path.abspath(args.bdir)
    if (args.kdecan is not None) and (args.ulg is not None):
        abs_kdecan_file = find_file_in_folder(
            f'{abs_bdir}/logs', '.kdecan', args.kdecan)
        abs_ulg_file = find_file_in_folder(
            f'{abs_bdir}/logs', '.ulg', args.ulg)
        print(f'kdecan file: {abs_kdecan_file}')
        print(f'ulg file: {abs_ulg_file}')

        ku_plot = KUPlot(abs_bdir)
        abs_ku_tag = f'kdecan_log_{args.kdecan}_ulg_log_{args.ulg}'
        ku_plot.process_file(abs_kdecan_file, abs_ulg_file, abs_ku_tag)
        exit(0)

        # abs_bdir = os.path.abspath(args.bdir)
        # ku_plot = KUPlot(abs_bdir)
        # abs_kdecan_file = ku_plot.find_file('.kdecan', args.kdecan)
        # abs_ulg_file = ku_plot.find_file('.ulg', args.ulg)
        # abs_ku_tag = f'kdecan_log_{args.kdecan}_ulg_log_{args.ulg}'
        # if os.path.isfile(abs_kdecan_file) and os.path.isfile(abs_ulg_file):
        #     ku_plot.process_file(
        #         abs_kdecan_file, abs_ulg_file, abs_ku_tag)
        # else:
        #     raise RuntimeError(
        #         f'No such files {abs_kdecan_file} or {abs_ulg_file}')

    print('Error parsing user arguments')
    print(f'firefly file: {args.firefly}')
    print(f'kdecan file: {args.kdecan}')
    print(f'ulg file: {args.ulg}')
    exit(0)

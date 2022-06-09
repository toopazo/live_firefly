import argparse

import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')
import os
import pandas

import numpy as np

from toopazo_tools.file_folder import FileFolderTools as FFTools
from firefly_parse_fu import FUParser, UlgParserTools as UlgPT
from firefly_database import FileTagData
from firefly_parse_keys import UlgDictKeys


class FUPlot:
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
        # self.figsize = (10, 6)
        self.img_cnt = 0

    def incr_img_cnt(self):
        self.img_cnt = self.img_cnt + 1
        return f'img{self.img_cnt}'

    def process_file(self, firefly_file, ulg_file, file_tag):
        fu_parser = FUParser(self.bdir, firefly_file, ulg_file)
        firefly_df = fu_parser.firefly_df
        ulg_dict = fu_parser.ulg_dict
        fu_data_dict = FileTagData.data_dict(file_tag)

        ulg_dict = UlgPT.synchronize_ulg_dict(ulg_dict, verbose=True)

        [firefly_df, ulg_dict] = FUParser.reset_index(firefly_df, ulg_dict)
        t_delay = fu_data_dict['t_delay']
        ulg_dict = FUParser.apply_delay_and_reset_index(ulg_dict, t_delay)

        ulg_out_df = ulg_dict[UlgDictKeys.ulg_out_df]
        new_index = ulg_out_df.index
        firefly_df = FUParser.resample_firefly_df(
            firefly_df, new_index, verbose=False)

        arm_df = FUParser.get_arm_df(firefly_df)

        a = np.arange(20)
        plt.plot(a)
        #plt.show()
        b = 0

        t_outgnd = fu_data_dict['t_outgnd']
        time0 = t_outgnd[0]
        time1 = t_outgnd[1]
        mask = (firefly_df.index > time0) & (firefly_df.index <= time1)
        mask = pandas.Series(mask)
        mask.index = firefly_df.index
        [firefly_df, arm_df, ulg_dict] = FUParser.filter_by_mask(
            mask, firefly_df, arm_df, ulg_dict)
        _ = t_delay, arm_df

        # We are now ready to start using (firefly_df, arm_df, ulg_dict)
        # print(firefly_df.keys())
        # print(ulg_dict.keys())
        # print(arm_df.keys())

        firefly_df = FUParser.calibrate_current(
            firefly_df, file_tag=file_tag, get_cur_0rpm=False)
        arm_df = FUParser.get_arm_df(firefly_df)
        # firefly_df = FUParse.check_current(firefly_df)
        # firefly_df = FUParser.smooth_current(firefly_df)

        return

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


def find_file_in_folder(fpath, extension, log_num):
    selected_file = ''
    file_arr = FFTools.get_file_arr(fpath=fpath, extension=extension)
    for file in file_arr:
        if log_num is not None:
            pattern1 = f'_{log_num}_'
            pattern2 = f'_{log_num}.'
            if (pattern1 in file) or (pattern2 in file):
                selected_file = os.path.abspath(file)
                break
    print(f'[find_file_in_folder] fpath {fpath}')
    print(f'[find_file_in_folder] extension {extension}')
    print(f'[find_file_in_folder] log_num {log_num}')
    print(f'[find_file_in_folder] selected_file {selected_file}')
    return selected_file


def run_analysis():
    parser = argparse.ArgumentParser(
        description='Parse, process and plot .kdecan files')
    parser.add_argument('--bdir', action='store', required=True,
                        help='Base directory of [logs, tmp, plots] folders')
    parser.add_argument('--ulg', action='store', required=True,
                        help='Specific log file number to process')
    parser.add_argument('--firefly', action='store', required=True,
                        help='Specific log file number to process')

    args = parser.parse_args()

    abs_bdir = os.path.abspath(args.bdir)
    if (args.firefly is not None) and (args.ulg is not None):
        abs_firefly_file = find_file_in_folder(
            f'{abs_bdir}/logs', '.firefly', args.firefly)
        abs_ulg_file = find_file_in_folder(
            f'{abs_bdir}/logs', '.ulg', args.ulg)
        print(f'firefly file: {abs_firefly_file}')
        print(f'ulg file: {abs_ulg_file}')

        fu_plot = FUPlot(abs_bdir)
        abs_fu_tag = f'firefly_log_{args.firefly}_ulg_log_{args.ulg}'
        fu_plot.process_file(abs_firefly_file, abs_ulg_file, abs_fu_tag)
        exit(0)

    print('Error parsing user arguments')
    print(f'firefly file: {args.firefly}')
    print(f'kdecan file: {args.kdecan}')
    print(f'ulg file: {args.ulg}')
    exit(0)


if __name__ == '__main__':
    run_analysis()

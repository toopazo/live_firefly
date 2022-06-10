import os
import pandas as pd
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
        self.img_cnt = 0

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
            firefly_df, new_index, verbose=True)

        arm_df = FUParser.get_arm_df(firefly_df)

        t_outgnd = fu_data_dict['t_outgnd']
        time0 = t_outgnd[0]
        time1 = t_outgnd[1]
        mask = (firefly_df.index > time0) & (firefly_df.index <= time1)
        mask = pd.Series(mask)
        mask.index = firefly_df.index
        [firefly_df, arm_df, ulg_dict] = FUParser.filter_by_mask(
            mask, firefly_df, arm_df, ulg_dict)
        _ = t_delay, arm_df

        firefly_df = FUParser.calibrate_current(
            firefly_df, file_tag=file_tag, get_cur_0rpm=False)
        arm_df = FUParser.get_arm_df(firefly_df)

        return firefly_df, arm_df, ulg_dict


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


def get_dfs(data_source, ulg, firefly):
    abs_dir = os.path.abspath(data_source)
    if (firefly is not None) and (ulg is not None):
        abs_firefly_file = find_file_in_folder(
            f'{abs_dir}/logs', '.firefly', firefly)
        abs_ulg_file = find_file_in_folder(
            f'{abs_dir}/logs', '.ulg', ulg)
        print(f'firefly file: {abs_firefly_file}')
        print(f'ulg file: {abs_ulg_file}')

        fu_plot = FUPlot(abs_dir)
        abs_fu_tag = f'firefly_log_{firefly}_ulg_log_{ulg}'
        return fu_plot.process_file(abs_firefly_file, abs_ulg_file, abs_fu_tag)

    print('Error parsing user arguments')
    print(f'firefly file: {firefly}')
    print(f'ulg file: {ulg}')
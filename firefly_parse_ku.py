import copy
import argparse
import pandas
import numpy as np
import os

from toopazo_tools.pandas import DataframeTools
from toopazo_tools.file_folder import FileFolderTools as FFTools
from live_esc.kde_uas85uvc.kdecan_parse import KdecanParser, EscidParserTools
from toopazo_ulg.parse_file import UlgParser, UlgParserTools


class KUParser:
    def __init__(self, bdir, kdecan_file, ulg_file):
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

        if not os.path.isfile(kdecan_file):
            raise RuntimeError(f'[KUParser] No such file {kdecan_file}')
        else:
            self.kdecan_file = kdecan_file
            self.escid_dict = KdecanParser.get_escid_dict(kdecan_file)

        if not os.path.isfile(ulg_file):
            raise RuntimeError(f'[KUParser] No such file {ulg_file}')
        else:
            self.ulg_file = ulg_file
            UlgParser.check_ulog2csv(self.tmpdir, self.ulg_file)
            self.ulg_dict = UlgParser.get_ulg_dict(self.tmpdir, ulg_file)

    @staticmethod
    def get_kdecan_arrays(kdecan_df, escid):
        assert isinstance(kdecan_df, pandas.DataFrame)
        df_tmp = kdecan_df[kdecan_df[KdecanParser.col_escid] == escid]
        assert isinstance(df_tmp, pandas.DataFrame)

        time = df_tmp.index
        current = df_tmp[KdecanParser.col_current].values
        voltage = df_tmp[KdecanParser.col_voltage].values
        angvel = df_tmp[KdecanParser.col_rpm].values
        throttle = df_tmp[KdecanParser.col_inthtl].values
        power = df_tmp[KdecanParser.col_voltage].values * df_tmp[
            KdecanParser.col_current].values

        data_arr = [time, current, voltage, angvel, throttle, power]
        return data_arr

    @staticmethod
    def filter_by_hover(escid_dict, ulg_dict):
        max_vnorm = 0.3

        ref_df = ulg_dict['ulg_out_df']
        ref_cond = ref_df[f'output[0]'] > 1200
        ulg_dict = UlgParserTools.remove_by_condition(ulg_dict, ref_cond)
        # Use same ref_cond for escid_dict
        escid_dict = EscidParserTools.remove_by_condition(escid_dict, ref_cond)

        ref_df = ulg_dict['ulg_pv_df']
        ref_cond = ref_df['vnorm'] < max_vnorm
        ulg_dict = UlgParserTools.remove_by_condition(ulg_dict, ref_cond)
        # Use same ref_cond for escid_dict
        escid_dict = EscidParserTools.remove_by_condition(escid_dict, ref_cond)

        ref_df = ulg_dict['ulg_in_df']
        ref_cond = ref_df['az cmd'] > 0.4
        ulg_dict = UlgParserTools.remove_by_condition(ulg_dict, ref_cond)
        # Use same ref_cond for escid_dict
        escid_dict = EscidParserTools.remove_by_condition(escid_dict, ref_cond)

        return copy.deepcopy([escid_dict, ulg_dict])

    @staticmethod
    def synchronize(escid_dict, ulg_dict):

        esc11_num_samples = escid_dict['esc11_df'].shape[0]
        ulg_num_samples = ulg_dict['ulg_out_df'].shape[0]
        if esc11_num_samples != ulg_num_samples:
            raise RuntimeError('esc11_num_samples != ulg_num_samples')

        ref_escid = 11
        min_throttle = 950

        ref_df = escid_dict[f'esc{ref_escid}_df']
        ref_cond = ref_df['inthtl us'] > min_throttle
        escid_dict = EscidParserTools.remove_by_condition(escid_dict, ref_cond)

        ref_df = ulg_dict['ulg_out_df']
        ref_cond = ref_df[f'output[{int(ref_escid - 11)}]'] > min_throttle
        ulg_dict = UlgParserTools.remove_by_condition(ulg_dict, ref_cond)

        escid_dict = DataframeTools.reset_index(escid_dict)
        ulg_dict = DataframeTools.reset_index(ulg_dict)

        # esc11_index = escid_dict['esc11_df'].index
        # ulg_index = ulg_dict['ulg_out_df'].index
        # for i in range(0, max(len(esc11_index), len(ulg_index))):
        #     if abs(esc11_index[i] - ulg_index[i]) > 10**-4:
        #         print(f'esc11_index[{i}] {esc11_index[i]}')
        #         print(f'ulg_index[{i}] {ulg_index[i]}')
        #         raise RuntimeError(
        #             'abs(esc11_index[i] - ulg_index[i]) > 10**-4')

        # print(f'esc11_index {esc11_index}')
        # print(f'ulg_index {ulg_index}')
        # print(f'index_diff {ulg_index-esc11_index}')

        # KUParser.calculate_lag_cost(
        #     esc11_df, ulg_out_df, delta=1, reference_escid=11)

        esc11_num_samples = escid_dict['esc11_df'].shape[0]
        ulg_num_samples = ulg_dict['ulg_out_df'].shape[0]

        if esc11_num_samples > ulg_num_samples:
            ref_df = escid_dict['esc11_df']
            escid_dict = DataframeTools.remove_by_index(
                escid_dict, ref_df.index[-1])

        if ulg_num_samples > esc11_num_samples:
            ref_df = ulg_dict['ulg_out_df']
            ulg_dict = DataframeTools.remove_by_index(
                ulg_dict, ref_df.index[-1])

        esc11_num_samples = escid_dict['esc11_df'].shape[0]
        ulg_num_samples = ulg_dict['ulg_out_df'].shape[0]
        if esc11_num_samples != ulg_num_samples:
            print(f'esc11_num_samples {esc11_num_samples}')
            print(f'ulg_num_samples {ulg_num_samples}')
            raise RuntimeError('esc11_num_samples != ulg_num_samples')

        return copy.deepcopy([escid_dict, ulg_dict])

    @staticmethod
    def calculate_lag_cost(escid_df, ulg_out_df, delta, reference_escid):
        assert isinstance(ulg_out_df, pandas.DataFrame)
        assert isinstance(escid_df, pandas.DataFrame)
        cost_arr = []
        delta_arr = np.arange(-delta, delta + 1)
        for delta in delta_arr:

            escid_throttle = escid_df['inthtl us'].values
            ulg_throttle = ulg_out_df[
                f'output[{int(reference_escid-11)}]'].values

            escid_len = len(escid_throttle)
            ulg_len = len(ulg_throttle)
            if ulg_len > escid_len:
                print(f'ulg_len {ulg_len} > escid_len {escid_len}')
            # smallest_indx = min(len(escid_throttle), len(ulg_throttle))

            escid_throttle = escid_throttle[delta:]
            ulg_val = ulg_throttle
            err = escid_throttle[delta:] - ulg_val
            # cost = np.sum(err * err)
            # cost = np.sum(np.abs(err))
            # cost = np.sum(err[int(smalles_indx*0.2):int(smalles_indx*0.8)])
            err = np.abs(err)
            # err1 = err[:int(smalles_indx * 0.1)]
            # err2 = err[int(smalles_indx * 0.9):]
            smallest_indx = -1
            err1 = err[:int(smallest_indx * 0.1)]
            err2 = err[:-int(smallest_indx * 0.1)]
            cost = np.sum(list(err1)+list(err2))
            # # print(int(smalles_indx * 0.01)) => 7
            # ib = int(smalles_indx/2) * 0
            # err = err[ib:ib+delta]
            # cost = np.sum(err)
            cost_arr.append(cost)
        return [delta_arr, cost_arr]

    @staticmethod
    def apply_kdecan_shift(kdecan_df, kdecan_shift):
        assert isinstance(kdecan_df, pandas.DataFrame)
        # print(kdecan_df)
        # print(kdecan_df.index)
        # print(kdecan_df.shape)
        # print(kdecan_df.shape[0])
        # print(kdecan_df.iloc[list(range(kdecan_shift, kdecan_df.shape[0]))])

        # kdecan_time[kdecan_shift:] - kdecan_time[kdecan_shift],
        # kdecan_throttle[kdecan_shift:]
        num_rows = kdecan_df.shape[0]
        kdecan_df = kdecan_df.iloc[list(range(kdecan_shift, num_rows))]
        num_rows_after = kdecan_df.shape[0]
        if num_rows_after == num_rows and kdecan_shift != 0:
            arg = f'num_rows_after {num_rows_after} == num_rows {num_rows}'
            raise RuntimeError(arg)
        return kdecan_df


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
        description='Parse, process and plot .ulg .kdecan and .firefly files')
    parser.add_argument('--bdir', action='store', required=True,
                        help='Base directory of [logs, tmp, plots] folders')
    parser.add_argument('--kdecan', action='store', required=False,
                        help='Specific log file number to process')
    parser.add_argument('--ulg', action='store', required=False,
                        help='Specific log file number to process')
    parser.add_argument('--firefly', action='store', required=False,
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

        ku_parser = KUParser(abs_bdir, abs_kdecan_file, abs_ulg_file)
        exit(0)

    print('Error parsing user arguments')
    print(f'firefly file: {args.firefly}')
    print(f'kdecan file: {args.kdecan}')
    print(f'ulg file: {args.ulg}')
    exit(0)

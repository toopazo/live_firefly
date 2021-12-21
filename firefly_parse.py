import copy
import datetime
import sys

import argparse
import pandas
import numpy as np
import os
from live_esc.kde_uas85uvc.kdecan_parse import KdecanParser
from toopazo_ulg.parse_file import UlgParser
from toopazo_tools.file_folder import FileFolderTools as FFTools


class FireflyParser:
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
            raise RuntimeError('Directories are not present or could not be created')

        if not os.path.isfile(kdecan_file):
            raise RuntimeError(f'[FireflyParser] No such file {kdecan_file}')
        else:
            self.kdecan_file = kdecan_file
            self.kdecan_df = KdecanParser.get_pandas_dataframe(kdecan_file, time_win=None)
        if not os.path.isfile(ulg_file):
            raise RuntimeError(f'[FireflyParser] No such file {ulg_file}')
        else:
            tmpdir = self.tmpdir
            self.ulg_file = ulg_file

            UlgParser.check_ulog2csv(self.tmpdir, self.ulg_file)

            df_pv = UlgParser.get_pandas_dataframe_pos_vel(tmpdir, ulg_file, time_win=None)
            self.ulg_pv_df = df_pv
            [df_att, df_attsp] = UlgParser.get_pandas_dataframe_rpy_angles(tmpdir, ulg_file, time_win=None)
            self.ulg_att_df = df_att
            self.ulg_attsp_df = df_attsp
            [df_angvel, df_angvelsp] = UlgParser.get_pandas_dataframe_pqr_angvel(tmpdir, ulg_file, time_win=None)
            self.ulg_angvel_df = df_angvel
            self.ulg_angvelsp_df = df_angvelsp
            [df_sticks, df_switches] = UlgParser.get_pandas_dataframe_man_ctrl(tmpdir, ulg_file, time_win=None)
            self.ulg_sticks_df = df_sticks
            self.ulg_switches_df = df_switches
            [df_in, df_out] = UlgParser.get_pandas_dataframe_ctrl_alloc(tmpdir, ulg_file, time_win=None)
            self.ulg_in_df = df_in
            self.ulg_out_df = df_out

            self.ulg_dict = {
                'ulg_pv_df': self.ulg_pv_df,
                # 'ulg_att_df': self.ulg_att_df,
                # 'ulg_attsp_df': self.ulg_attsp_df,
                # 'ulg_angvel_df': self.ulg_angvel_df,
                # 'ulg_angvelsp_df': self.ulg_angvelsp_df,
                # 'ulg_sticks_df': self.ulg_sticks_df,
                # 'ulg_switches_df': self.ulg_switches_df,
                'ulg_in_df': self.ulg_in_df,
                'ulg_out_df': self.ulg_out_df,
            }

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
        power = df_tmp[KdecanParser.col_voltage].values * df_tmp[KdecanParser.col_current].values

        data_arr = [time, current, voltage, angvel, throttle, power]
        return data_arr

    @staticmethod
    def get_kdecan_arm_dict(escid_dict):
        esc11_df = escid_dict['esc11_df']
        esc12_df = escid_dict['esc12_df']
        esc13_df = escid_dict['esc13_df']
        esc14_df = escid_dict['esc14_df']
        esc15_df = escid_dict['esc15_df']
        esc16_df = escid_dict['esc16_df']
        esc17_df = escid_dict['esc17_df']
        esc18_df = escid_dict['esc18_df']

        esc11_power = esc11_df[KdecanParser.col_voltage].values * esc11_df[KdecanParser.col_current].values
        esc12_power = esc12_df[KdecanParser.col_voltage].values * esc12_df[KdecanParser.col_current].values
        esc13_power = esc13_df[KdecanParser.col_voltage].values * esc13_df[KdecanParser.col_current].values
        esc14_power = esc14_df[KdecanParser.col_voltage].values * esc14_df[KdecanParser.col_current].values
        esc15_power = esc15_df[KdecanParser.col_voltage].values * esc15_df[KdecanParser.col_current].values
        esc16_power = esc16_df[KdecanParser.col_voltage].values * esc16_df[KdecanParser.col_current].values
        esc17_power = esc17_df[KdecanParser.col_voltage].values * esc17_df[KdecanParser.col_current].values
        esc18_power = esc18_df[KdecanParser.col_voltage].values * esc18_df[KdecanParser.col_current].values

        arm1_power = esc11_power + esc16_power
        arm2_power = esc12_power + esc15_power
        arm3_power = esc13_power + esc18_power
        arm4_power = esc14_power + esc17_power

        arm1_eta_angvel = esc11_df[KdecanParser.col_rpm].values / esc16_df[KdecanParser.col_rpm].values
        arm2_eta_angvel = esc12_df[KdecanParser.col_rpm].values / esc15_df[KdecanParser.col_rpm].values
        arm3_eta_angvel = esc13_df[KdecanParser.col_rpm].values / esc18_df[KdecanParser.col_rpm].values
        arm4_eta_angvel = esc14_df[KdecanParser.col_rpm].values / esc17_df[KdecanParser.col_rpm].values

        arm1_eta_throttle = esc11_df[KdecanParser.col_inthtl].values / esc16_df[KdecanParser.col_inthtl].values
        arm2_eta_throttle = esc12_df[KdecanParser.col_inthtl].values / esc15_df[KdecanParser.col_inthtl].values
        arm3_eta_throttle = esc13_df[KdecanParser.col_inthtl].values / esc18_df[KdecanParser.col_inthtl].values
        arm4_eta_throttle = esc14_df[KdecanParser.col_inthtl].values / esc17_df[KdecanParser.col_inthtl].values

        # Ignore unused arrays
        # _ = esc11_time + esc12_time + esc13_time + esc14_time + \
        #     esc15_time + esc16_time + esc17_time + esc18_time
        # _ = esc11_current + esc12_current + esc13_current + esc14_current + \
        #     esc15_current + esc16_current + esc17_current + esc18_current
        # _ = esc11_voltage + esc12_voltage + esc13_voltage + esc14_voltage + \
        #     esc15_voltage + esc16_voltage + esc17_voltage + esc18_voltage

        arm_dict = {
            'arm1': [arm1_power, arm1_eta_angvel, arm1_eta_throttle],
            'arm2': [arm2_power, arm2_eta_angvel, arm2_eta_throttle],
            'arm3': [arm3_power, arm3_eta_angvel, arm3_eta_throttle],
            'arm4': [arm4_power, arm4_eta_angvel, arm4_eta_throttle],
        }
        return arm_dict

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
    def kdecan_to_escid_dict(kdecan_df):
        # Create new dataframes for each escid
        esc11_df = FireflyParser.kdecan_to_escid_dataframe(kdecan_df, escid=11)
        esc12_df = FireflyParser.kdecan_to_escid_dataframe(kdecan_df, escid=12)
        esc13_df = FireflyParser.kdecan_to_escid_dataframe(kdecan_df, escid=13)
        esc14_df = FireflyParser.kdecan_to_escid_dataframe(kdecan_df, escid=14)
        esc15_df = FireflyParser.kdecan_to_escid_dataframe(kdecan_df, escid=15)
        esc16_df = FireflyParser.kdecan_to_escid_dataframe(kdecan_df, escid=16)
        esc17_df = FireflyParser.kdecan_to_escid_dataframe(kdecan_df, escid=17)
        esc18_df = FireflyParser.kdecan_to_escid_dataframe(kdecan_df, escid=18)
        min_num_samples = min(
            esc11_df.shape[0], esc12_df.shape[0],
            esc13_df.shape[0], esc14_df.shape[0],
            esc15_df.shape[0], esc16_df.shape[0],
            esc17_df.shape[0], esc18_df.shape[0]
        )
        escid_dict = {
            'esc11_df': esc11_df.iloc[:min_num_samples],
            'esc12_df': esc12_df.iloc[:min_num_samples],
            'esc13_df': esc13_df.iloc[:min_num_samples],
            'esc14_df': esc14_df.iloc[:min_num_samples],
            'esc15_df': esc15_df.iloc[:min_num_samples],
            'esc16_df': esc16_df.iloc[:min_num_samples],
            'esc17_df': esc17_df.iloc[:min_num_samples],
            'esc18_df': esc18_df.iloc[:min_num_samples],
        }
        return copy.deepcopy(escid_dict)

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

        # FireflyParser.calculate_lag_cost(
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
            ulg_throttle = ulg_out_df[f'output[{int(reference_escid-11)}]'].values

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
    def kdecan_to_escid_dataframe(kdecan_df, escid):
        assert isinstance(kdecan_df, pandas.DataFrame)
        escid_cond = kdecan_df['escid'] == escid
        escid_df = kdecan_df[escid_cond]

        # escid_t0 = escid_df.index[0]
        # escid_tf = escid_df.index[-1]
        # escid_num_samples = len(escid_df.index)
        # escid_sample_period = escid_tf / escid_num_samples
        # print(f'escid {escid}')
        # print(f'escid_t0 {escid_t0}')
        # print(f'escid_tf {escid_tf}')
        # print(f'escid_num_samples {escid_num_samples}')
        # print(f'escid_sample_period {escid_sample_period}')

        data = {
            'voltage V': escid_df['voltage V'].values,
            'current A': escid_df['current A'].values,
            'angVel rpm': escid_df['angVel rpm'].values,
            'temp degC': escid_df['temp degC'].values,
            'warning': escid_df['warning'].values,
            'inthtl us': escid_df['inthtl us'].values,
            'outthtl perc': escid_df['outthtl perc'].values,
        }
        index = escid_df.index
        new_escid_df = pandas.DataFrame(data=data, index=index)

        # new_escid_t0 = new_escid_df.index[0]
        # new_escid_tf = new_escid_df.index[-1]
        # new_escid_num_samples = len(new_escid_df.index)
        # new_escid_sample_period = new_escid_tf / new_escid_num_samples
        # print(f'new_escid_t0 {new_escid_t0}')
        # print(f'new_escid_tf {new_escid_tf}')
        # print(f'new_escid_num_samples {new_escid_num_samples}')
        # print(f'new_escid_sample_period {new_escid_sample_period}')

        return copy.deepcopy(new_escid_df)

    @staticmethod
    def apply_kdecan_shift(kdecan_df, kdecan_shift):
        assert isinstance(kdecan_df, pandas.DataFrame)
        # print(kdecan_df)
        # print(kdecan_df.index)
        # print(kdecan_df.shape)
        # print(kdecan_df.shape[0])
        # print(kdecan_df.iloc[list(range(kdecan_shift, kdecan_df.shape[0]))])

        # kdecan_time[kdecan_shift:]-kdecan_time[kdecan_shift], kdecan_throttle[kdecan_shift:]
        num_rows = kdecan_df.shape[0]
        kdecan_df = kdecan_df.iloc[list(range(kdecan_shift, num_rows))]
        num_rows_after = kdecan_df.shape[0]
        if num_rows_after == num_rows and kdecan_shift != 0:
            raise RuntimeError(f'num_rows_after {num_rows_after} == num_rows {num_rows}')
        return kdecan_df


class EscidParserTools:
    @staticmethod
    def resample(escid_dict, time_secs, max_delta):
        if DataframeTools.check_time_difference(escid_dict, max_delta):
            # time_secs = DataframeTools.shortest_time_secs(escid_dict)
            pass
        else:
            raise RuntimeError('EscidParserTools.check_time_difference failed')

        new_escid_dict = {}
        x = time_secs
        for key, escid_df in escid_dict.items():
            # xp = escid_df.index
            xp = DataframeTools.index_to_elapsed_time(escid_df)
            data = {
                'voltage V':  np.interp(x, xp, fp=escid_df['voltage V']),
                'current A':  np.interp(x, xp, fp=escid_df['current A']),
                'angVel rpm': np.interp(x, xp, fp=escid_df['angVel rpm']),
                'temp degC': np.interp(x, xp, fp=escid_df['temp degC']),
                'inthtl us': np.interp(x, xp, fp=escid_df['inthtl us']),
                'outthtl perc': np.interp(x, xp, fp=escid_df['outthtl perc']),
            }
            index = x
            new_escid_df = pandas.DataFrame(data=data, index=index)
            new_escid_dict[key] = new_escid_df
            # print(f"key {key} ------------------------")
            # print(f"{escid_df}")
            # print(f"{new_escid_df}")
        return copy.deepcopy(new_escid_dict)

    @staticmethod
    def synchronize(escid_dict, time_secs):
        max_delta = 0.01
        if DataframeTools.check_time_difference(escid_dict, max_delta):
            # time_secs = DataframeTools.shortest_time_secs(escid_dict)
            new_escid_dict = EscidParserTools.resample(
                escid_dict, time_secs, max_delta)
            return copy.deepcopy(new_escid_dict)
        else:
            raise RuntimeError('EscidParserTools.check_time_difference failed')

    @staticmethod
    def remove_by_condition(escid_dict, escid_ref_cond):
        # assert isinstance(kdecan_df, pandas.DataFrame)
        # assert isinstance(ulg_out_df, pandas.DataFrame)

        esc11_df = escid_dict['esc11_df']
        esc12_df = escid_dict['esc12_df']
        esc13_df = escid_dict['esc13_df']
        esc14_df = escid_dict['esc14_df']
        esc15_df = escid_dict['esc15_df']
        esc16_df = escid_dict['esc16_df']
        esc17_df = escid_dict['esc17_df']
        esc18_df = escid_dict['esc18_df']

        # escid_df = escid_dict[f'esc{reference_escid}_df']
        # escid_ref_cond = escid_df['inthtl us'] > min_throttle

        escid_ref_cond.index = esc11_df.index
        esc11_df = esc11_df[escid_ref_cond]
        escid_ref_cond.index = esc12_df.index
        esc12_df = esc12_df[escid_ref_cond]
        escid_ref_cond.index = esc13_df.index
        esc13_df = esc13_df[escid_ref_cond]
        escid_ref_cond.index = esc14_df.index
        esc14_df = esc14_df[escid_ref_cond]
        escid_ref_cond.index = esc15_df.index
        esc15_df = esc15_df[escid_ref_cond]
        escid_ref_cond.index = esc16_df.index
        esc16_df = esc16_df[escid_ref_cond]
        escid_ref_cond.index = esc17_df.index
        esc17_df = esc17_df[escid_ref_cond]
        escid_ref_cond.index = esc18_df.index
        esc18_df = esc18_df[escid_ref_cond]

        escid_dict = {
            'esc11_df': esc11_df,
            'esc12_df': esc12_df,
            'esc13_df': esc13_df,
            'esc14_df': esc14_df,
            'esc15_df': esc15_df,
            'esc16_df': esc16_df,
            'esc17_df': esc17_df,
            'esc18_df': esc18_df,
        }

        return copy.deepcopy(escid_dict)


class UlgParserTools:
    @staticmethod
    def synchronize(ulg_dict, time_secs):
        max_delta = 0.01
        if DataframeTools.check_time_difference(ulg_dict, max_delta):
            # time_secs = DataframeTools.shortest_time_secs(ulg_dict)
            new_df_arr = UlgParserTools.resample(
                ulg_dict, time_secs, max_delta)
            return copy.deepcopy(new_df_arr)
        else:
            raise RuntimeError('UlgParserTools.check_time_difference failed')

    @staticmethod
    def resample(escid_dict, time_secs, max_delta):
        if DataframeTools.check_time_difference(escid_dict, max_delta):
            # time_secs = DataframeTools.shortest_time_secs(escid_dict)
            pass
        else:
            raise RuntimeError('EscidParserTools.check_time_difference failed')

        new_escid_dict = {}
        x = time_secs
        for key, ulg_df in escid_dict.items():
            # xp = ulg_df.index
            xp = DataframeTools.index_to_elapsed_time(ulg_df)
            data = UlgParserTools.get_data_by_type(x, xp, ulg_df, key)
            index = x
            new_escid_df = pandas.DataFrame(data=data, index=index)
            new_escid_dict[key] = new_escid_df
            # print(f"key {key} ------------------------")
            # print(f"{ulg_df}")
            # print(f"{new_escid_df}")
        return copy.deepcopy(new_escid_dict)

    @staticmethod
    def get_data_by_type(x, xp, ulg_df, ulg_type):
        if ulg_type == 'ulg_pv_df':
            data = {
                'x': np.interp(x, xp, fp=ulg_df['x']),
                'y': np.interp(x, xp, fp=ulg_df['y']),
                'z': np.interp(x, xp, fp=ulg_df['z']),
                'vx': np.interp(x, xp, fp=ulg_df['vx']),
                'vy': np.interp(x, xp, fp=ulg_df['vy']),
                'vz': np.interp(x, xp, fp=ulg_df['vz']),
                'vnorm': np.interp(x, xp, fp=ulg_df['vnorm']),
                'pnorm': np.interp(x, xp, fp=ulg_df['pnorm']),
            }
            return data
        if ulg_type == 'ulg_att_df':
            data = {
                'roll': np.interp(x, xp, fp=ulg_df['roll']),
                'pitch': np.interp(x, xp, fp=ulg_df['pitch']),
                'yaw': np.interp(x, xp, fp=ulg_df['yaw']),
            }
            return data
        if ulg_type == 'ulg_in_df':
            data = {
                'roll rate cmd': np.interp(x, xp, fp=ulg_df['roll rate cmd']),
                'pitch rate cmd': np.interp(x, xp, fp=ulg_df['pitch rate cmd']),
                'yaw rate cmd': np.interp(x, xp, fp=ulg_df['yaw rate cmd']),
                'az cmd': np.interp(x, xp, fp=ulg_df['az cmd']),
            }
            return data
        if ulg_type == 'ulg_out_df':
            data = {
                'output[0]': np.interp(x, xp, fp=ulg_df['output[0]']),
                'output[1]': np.interp(x, xp, fp=ulg_df['output[1]']),
                'output[2]': np.interp(x, xp, fp=ulg_df['output[2]']),
                'output[3]': np.interp(x, xp, fp=ulg_df['output[3]']),
                'output[4]': np.interp(x, xp, fp=ulg_df['output[4]']),
                'output[5]': np.interp(x, xp, fp=ulg_df['output[5]']),
                'output[6]': np.interp(x, xp, fp=ulg_df['output[6]']),
                'output[7]': np.interp(x, xp, fp=ulg_df['output[7]']),
            }
            return data

    @staticmethod
    def remove_by_condition(ulg_dict, ulg_ref_cond):
        ulg_pv_df = ulg_dict['ulg_pv_df']
        ulg_in_df = ulg_dict['ulg_in_df']
        ulg_out_df = ulg_dict['ulg_out_df']

        # ulg_key = f'output[{int(reference_escid - 11)}]'
        # ulg_ref_cond = ulg_out_df[ulg_key] > min_throttle

        ulg_ref_cond.index = ulg_pv_df.index
        ulg_pv_df = ulg_pv_df[ulg_ref_cond]
        ulg_ref_cond.index = ulg_in_df.index
        ulg_in_df = ulg_in_df[ulg_ref_cond]
        ulg_ref_cond.index = ulg_out_df.index
        ulg_out_df = ulg_out_df[ulg_ref_cond]

        ulg_dict = {
            'ulg_pv_df': ulg_pv_df,
            # 'ulg_att_df': ulg_att_df,
            # 'ulg_attsp_df': ulg_attsp_df,
            # 'ulg_angvel_df': ulg_angvel_df,
            # 'ulg_angvelsp_df': ulg_angvelsp_df,
            # 'ulg_sticks_df': ulg_sticks_df,
            # 'ulg_switches_df': ulg_switches_df,
            'ulg_in_df': ulg_in_df,
            'ulg_out_df': ulg_out_df,
        }
        return copy.deepcopy(ulg_dict)


class DataframeTools:
    @staticmethod
    def reset_index(df_dict):
        assert isinstance(df_dict, dict)
        for key, val in df_dict.items():
            time_secs = DataframeTools.index_to_elapsed_time(val)
            df_dict[key].index = time_secs
        return copy.deepcopy(df_dict)

    @staticmethod
    def timedelta_to_float(time_arr):
        # Just in case, convert to secs
        time_arr = np.array(
            time_arr).astype("timedelta64[ms]").astype(int) / 1000
        return time_arr

    @staticmethod
    def index_to_elapsed_time(dataframe):
        if isinstance(dataframe.index, pandas.DatetimeIndex):
            time_delta = dataframe.index - dataframe.index[0]
            time_secs = DataframeTools.timedelta_to_float(time_delta)
        else:
            time_secs = dataframe.index - dataframe.index[0]
        return time_secs

    @staticmethod
    def check_time_difference(df_coll, max_delta):
        df_arr = []
        if isinstance(df_coll, dict):
            for key, val in df_coll.items():
                df_arr.append(val)
        if isinstance(df_coll, list):
            df_arr = df_coll

        time_0_arr = []
        time_1_arr = []
        for df in df_arr:
            time_0_arr.append(df.index[0])
            time_1_arr.append(df.index[1])
        time_0_diff = np.diff(time_0_arr)
        time_1_diff = np.diff(time_1_arr)
        # Just in case, convert to secs
        time_0_diff = DataframeTools.timedelta_to_float(time_0_diff)
        time_1_diff = DataframeTools.timedelta_to_float(time_1_diff)
        # Convert to abs values
        time_0_diff = np.abs(time_0_diff)
        time_1_diff = np.abs(time_1_diff)
        if max(time_0_diff) > max_delta:
            return False
        if max(time_1_diff) > max_delta:
            return False
        return True

    @staticmethod
    def shortest_time_secs(df_coll):
        df_arr = []
        if isinstance(df_coll, dict):
            for key, val in df_coll.items():
                df_arr.append(val)
        if isinstance(df_coll, list):
            df_arr = df_coll

        time_1_arr = []
        time_secs_arr = []
        for df in df_arr:
            time_secs = DataframeTools.index_to_elapsed_time(df)
            time_1_arr.append(time_secs[-1])
            time_secs_arr.append(time_secs)
        # time_secs of the escid that has the samllest time_1
        i_smallest_time_1 = np.argmin(time_1_arr)
        i_time_secs = time_secs_arr[i_smallest_time_1]
        return i_time_secs

    @staticmethod
    def remove_by_index(df_dict, rm_index):
        for key, df in df_dict.items():
            assert isinstance(df, pandas.DataFrame)
            df_dict[key] = df.drop(rm_index)
        return copy.deepcopy(df_dict)


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
    abs_kdecan_file = find_file_in_folder(f'{abs_bdir}/logs', '.kdecan', args.kdecan)
    abs_ulg_file = find_file_in_folder(f'{abs_bdir}/logs', '.ulg', args.ulg)

    firefly_parser = FireflyParser(abs_bdir, abs_kdecan_file, abs_ulg_file)

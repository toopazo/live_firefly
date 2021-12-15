import copy
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
    def match_kdecan_and_ulg_dataframe(kdecan_df, ulg_out_df):
        assert isinstance(kdecan_df, pandas.DataFrame)
        assert isinstance(ulg_out_df, pandas.DataFrame)
        kdecan_num_rows = kdecan_df.shape[0]
        ulg_num_rows = ulg_out_df.shape[0]
        if kdecan_num_rows < ulg_num_rows:
            raise RuntimeError(f'kdecan_num_rows {kdecan_num_rows} < ulg_num_rows {ulg_num_rows}')

        # Reset times to start at zero second
        [kdecan_df, ulg_out_df] = FireflyParser.reset_kdecan_ulg_dataframe(kdecan_df, ulg_out_df)
        # Remove disarmed throttle
        [kdecan_df, ulg_out_df] = FireflyParser.remove_min_throttle(
            kdecan_df, ulg_out_df, min_throttle=950, reference_escid=11)

        # Create new dataframes for each escid
        esc11_df = FireflyParser.kdecan_to_escid_dataframe(kdecan_df, escid=11, reset_time=True)
        esc12_df = FireflyParser.kdecan_to_escid_dataframe(kdecan_df, escid=12, reset_time=True)
        esc13_df = FireflyParser.kdecan_to_escid_dataframe(kdecan_df, escid=13, reset_time=True)
        esc14_df = FireflyParser.kdecan_to_escid_dataframe(kdecan_df, escid=14, reset_time=True)
        esc15_df = FireflyParser.kdecan_to_escid_dataframe(kdecan_df, escid=15, reset_time=True)
        esc16_df = FireflyParser.kdecan_to_escid_dataframe(kdecan_df, escid=16, reset_time=True)
        esc17_df = FireflyParser.kdecan_to_escid_dataframe(kdecan_df, escid=17, reset_time=True)
        esc18_df = FireflyParser.kdecan_to_escid_dataframe(kdecan_df, escid=18, reset_time=True)
        min_num_samples = min(
            esc11_df.shape[0], esc12_df.shape[0], esc13_df.shape[0], esc14_df.shape[0],
            esc15_df.shape[0], esc16_df.shape[0], esc17_df.shape[0], esc18_df.shape[0]
        )
        escid_dict = {
            # 'ulg_out_df': ulg_out_df,
            # 'kdecan_df': kdecan_df,
            'esc11_df': esc11_df.iloc[:min_num_samples],
            'esc12_df': esc12_df.iloc[:min_num_samples],
            'esc13_df': esc13_df.iloc[:min_num_samples],
            'esc14_df': esc14_df.iloc[:min_num_samples],
            'esc15_df': esc15_df.iloc[:min_num_samples],
            'esc16_df': esc16_df.iloc[:min_num_samples],
            'esc17_df': esc17_df.iloc[:min_num_samples],
            'esc18_df': esc18_df.iloc[:min_num_samples],
        }

        [kdecan_df, ulg_out_df, escid_dict] = FireflyParser.remove_min_throttle_v2(
            kdecan_df, ulg_out_df, escid_dict, min_throttle=950, reference_escid=11)
        # FireflyParser.calculate_lag_cost(esc11_df, ulg_out_df, delta=1, reference_escid=11)

        return copy.deepcopy([kdecan_df, ulg_out_df, escid_dict])

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
    def remove_min_throttle_v2(kdecan_df, ulg_out_df, escid_dict, min_throttle, reference_escid):
        assert isinstance(kdecan_df, pandas.DataFrame)
        assert isinstance(ulg_out_df, pandas.DataFrame)

        esc11_df = escid_dict['esc11_df']
        esc12_df = escid_dict['esc12_df']
        esc13_df = escid_dict['esc13_df']
        esc14_df = escid_dict['esc14_df']
        esc15_df = escid_dict['esc15_df']
        esc16_df = escid_dict['esc16_df']
        esc17_df = escid_dict['esc17_df']
        esc18_df = escid_dict['esc18_df']

        # print(esc11_df.shape)
        # print(esc12_df.shape)
        # print(esc13_df.shape)
        # print(esc14_df.shape)
        # print(esc15_df.shape)
        # print(esc16_df.shape)
        # print(esc17_df.shape)
        # print(esc18_df.shape)
        # print(esc17_df)
        # print(esc18_df)

        escid_df = escid_dict[f'esc{reference_escid}_df']
        reference_cond = escid_df['inthtl us'] > min_throttle

        reference_cond.index = esc11_df.index
        esc11_df = esc11_df[reference_cond]
        reference_cond.index = esc12_df.index
        esc12_df = esc12_df[reference_cond]
        reference_cond.index = esc13_df.index
        esc13_df = esc13_df[reference_cond]
        reference_cond.index = esc14_df.index
        esc14_df = esc14_df[reference_cond]
        reference_cond.index = esc15_df.index
        esc15_df = esc15_df[reference_cond]
        reference_cond.index = esc16_df.index
        esc16_df = esc16_df[reference_cond]
        reference_cond.index = esc17_df.index
        esc17_df = esc17_df[reference_cond]
        reference_cond.index = esc18_df.index
        esc18_df = esc18_df[reference_cond]

        kdecan_df = kdecan_df[kdecan_df['escid'] == reference_escid]
        ulg_out_df = ulg_out_df[ulg_out_df[f'output[{int(reference_escid - 11)}]'] > min_throttle]

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
        return copy.deepcopy([kdecan_df, ulg_out_df, escid_dict])

    @staticmethod
    def remove_min_throttle(kdecan_df, ulg_out_df, min_throttle, reference_escid):
        assert isinstance(kdecan_df, pandas.DataFrame)
        assert isinstance(ulg_out_df, pandas.DataFrame)

        kdecan_df = kdecan_df[kdecan_df['inthtl us'] > min_throttle]
        ulg_out_df = ulg_out_df[ulg_out_df[f'output[{int(reference_escid-11)}]'] > min_throttle]
        # indx_after_ulg_tf = np.argwhere(kdecan_time_reset > ulg_tf)
        # print(f'indx_after_ulg_tf {indx_after_ulg_tf}')
        # print(f'kdecan_time_reset[indx_after_ulg_tf] {kdecan_time_reset[indx_after_ulg_tf]}')

        return [copy.deepcopy(kdecan_df), copy.deepcopy(ulg_out_df)]

    @staticmethod
    def reset_kdecan_ulg_dataframe(kdecan_df, ulg_out_df):
        assert isinstance(kdecan_df, pandas.DataFrame)
        assert isinstance(ulg_out_df, pandas.DataFrame)
        # Reset times to start at zero second
        ulg_time_reset = ulg_out_df.index - ulg_out_df.index[0]
        ulg_t0 = ulg_time_reset[0]
        ulg_tf = ulg_time_reset[-1]
        ulg_num_samples = len(ulg_time_reset)
        ulg_sample_period = ulg_tf / ulg_num_samples
        ulg_out_df.index = ulg_time_reset
        # print(f'ulg_time_reset {ulg_time_reset}')
        # print(ulg_out_df)
        print(f'ulg_t0 {ulg_t0}')
        print(f'ulg_tf {ulg_tf}')
        print(f'ulg_num_samples {ulg_num_samples}')
        print(f'ulg_sample_period {ulg_sample_period}')

        kdecan_time_reset = kdecan_df.index - kdecan_df.index[0]
        kdecan_time_reset = np.array(kdecan_time_reset.total_seconds())
        kdecan_t0 = kdecan_time_reset[0]
        kdecan_tf = kdecan_time_reset[-1]
        kdecan_num_samples = len(kdecan_time_reset)
        kdecan_sample_period = kdecan_tf / kdecan_num_samples
        kdecan_df.index = kdecan_time_reset
        # print(f'kdecan_time_reset {kdecan_time_reset}')
        # print(kdecan_df)
        print(f'kdecan_t0 {kdecan_t0}')
        print(f'kdecan_tf {kdecan_tf}')
        print(f'kdecan_num_samples {kdecan_num_samples}')
        print(f'kdecan_sample_period {kdecan_sample_period}')

        return [copy.deepcopy(kdecan_df), copy.deepcopy(ulg_out_df)]

    @staticmethod
    def kdecan_to_escid_dataframe(kdecan_df, escid, reset_time):
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
        if reset_time:
            index_arr = escid_df.index - escid_df.index[0]
        else:
            index_arr = escid_df.index
        new_escid_df = pandas.DataFrame(data, index=index_arr)

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

    # def get_all_pandas_dataframe(self):
    #     dataframe_dict = {
    #         'kdecan_df': self.kdecan_df,
    #         'ulg_pv_df': self.ulg_pv_df,
    #         'ulg_att_df': self.ulg_att_df,
    #         'ulg_attsp_df': self.ulg_attsp_df,
    #         'ulg_angvel_df': self.ulg_angvel_df,
    #         'ulg_angvelsp_df': self.ulg_angvelsp_df,
    #         'ulg_sticks_df': self.ulg_sticks_df,
    #         'ulg_switches_df': self.ulg_switches_df,
    #         'ulg_in_df': self.ulg_in_df,
    #         'ulg_out_df': self.ulg_out_df,
    #     }
    #     return dataframe_dict

    # def calculate_kdecan_shift(self, selected_escid, save_plot):
    #     [num_esc, kdecan_dict, ulg_dict] = self.get_time_and_throttle()
    #
    #     ax1 = None
    #     ax2 = None
    #     ax3 = None
    #     if save_plot:
    #         fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=self.figsize)
    #         ax1.grid(True)
    #         ax1.set_ylabel("correlation")
    #         ax1.set_xlabel("lags")
    #         ax2.grid(True)
    #         ax2.set_ylabel("throttle us")
    #         ax2.set_xlabel("time s")
    #         ax3.grid(True)
    #         ax3.set_ylabel("throttle us")
    #         ax3.set_xlabel("time s")
    #
    #     argmax_corr_arr = []
    #     lags = []
    #     for i in range(0, num_esc):
    #         escid = 10 + i + 1
    #         escid_key = f'escid_{escid}'
    #
    #         kdecan_time = kdecan_dict[escid_key][0]
    #         kdecan_throttle = kdecan_dict[escid_key][1]
    #         ulg_time = ulg_dict[escid_key][0]
    #         ulg_throttle = ulg_dict[escid_key][1]
    #
    #         # plt.plot(kdecan_time, kdecan_throttle)
    #         # plt.plot(ulg_time, ulg_throttle)
    #
    #         corr = signal.correlate(in1=ulg_throttle, in2=kdecan_throttle, mode='full', method='auto')
    #         lags = signal.correlation_lags(len(ulg_throttle), len(kdecan_throttle))
    #         corr = corr / np.max(corr)
    #         argmax_corr = np.argmax(corr)
    #         argmax_corr_arr.append(argmax_corr)
    #         lag_argmax_corr = lags[argmax_corr]
    #         # print(f'argmax_corr {argmax_corr}')
    #         # print(f'lag_argmax_corr {lag_argmax_corr}')
    #
    #         if save_plot:
    #             ax1.plot(lags, corr)
    #             ax2.plot(kdecan_time, kdecan_throttle)
    #             ax2.plot(ulg_time, ulg_throttle)
    #
    #     # argmax_corr = int(round(argmax_corr / num_esc, 0))
    #     argmax_corr = int(np.median(np.array(argmax_corr_arr)))
    #     lag_argmax_corr = lags[argmax_corr]
    #     # print(f'argmax_corr {argmax_corr}')
    #     # print(f'lag_argmax_corr {lag_argmax_corr}')
    #
    #     kdecan_shift = 0
    #     for i in range(0, num_esc):
    #         escid = 10 + i + 1
    #         escid_key = f'escid_{escid}'
    #
    #         kdecan_time = kdecan_dict[escid_key][0]
    #         kdecan_throttle = kdecan_dict[escid_key][1]
    #         ulg_time = ulg_dict[escid_key][0]
    #         ulg_throttle = ulg_dict[escid_key][1]
    #
    #         [delta_arr, cost_arr] = self.calculate_lag_cost(escid=selected_escid, lag=lag_argmax_corr, delta=10)
    #         argmin_cost = np.argmin(cost_arr)
    #         delta_argmin = delta_arr[argmin_cost]
    #         # print(f'delta_arr {delta_arr}')
    #         # print(f'cost_arr {cost_arr}')
    #         # print(f'delta_argmin {delta_argmin}')
    #
    #         kdecan_shift = -lag_argmax_corr + delta_argmin + 4
    #         if save_plot:
    #             ax3.plot(kdecan_time[kdecan_shift:]-kdecan_time[kdecan_shift], kdecan_throttle[kdecan_shift:])
    #             ax3.plot(ulg_time, ulg_throttle)
    #
    #     if save_plot:
    #         ax1.set_title(f'ulg to kdecan cross-correlation: lag_macorr {lag_argmax_corr}', fontsize=16)
    #
    #         pltname = f'firefly_kdecan_ulg.png'
    #         jpgfilename = os.path.abspath(f'{self.plotdir}/{pltname}')
    #         plt.savefig(jpgfilename)
    #
    #     return kdecan_shift

    # def get_time_and_throttle(self):
    #     kdecan_df = self.firefly_parser.kdecan_df
    #     assert isinstance(kdecan_df, pandas.DataFrame)
    #     # print(kdecan_df)
    #     kdecan_thtl_dict = {}
    #     for i in range(0, 8):
    #         escid = 10 + i + 1
    #         df_i = kdecan_df[kdecan_df['escid'] == escid]
    #         kdecan_thtl_dict[f'escid_{escid}'] = copy.deepcopy(df_i['inthtl us'])
    #         # print(kdecan_thtl_dict[f'escid_{escid}'])
    #
    #     ulg_out_df = self.firefly_parser.ulg_out_df
    #     assert isinstance(ulg_out_df, pandas.DataFrame)
    #     # print(ulg_out_df)
    #     ulg_thtl_dict = {}
    #     for i in range(0, 8):
    #         escid = 10 + i + 1
    #         df_i = ulg_out_df[f'output[{i}]']
    #         ulg_thtl_dict[f'escid_{escid}'] = copy.deepcopy(df_i)
    #         # print(ulg_thtl_dict[f'escid_{escid}'])
    #
    #     kdecan_dict = {}
    #     ulg_dict = {}
    #
    #     num_esc = 8
    #     for i in range(0, num_esc):
    #         escid = 10 + i + 1
    #         escid_key = f'escid_{escid}'
    #
    #         kdecan_thtl = kdecan_thtl_dict[escid_key]
    #         kdecan_time = kdecan_thtl.index - kdecan_thtl.index[0]
    #         kdecan_time = np.array(kdecan_time.total_seconds())
    #         kdecan_throttle = np.array(kdecan_thtl.values)
    #         kdecan_dict[escid_key] = copy.deepcopy([kdecan_time, kdecan_throttle])
    #         # print(type(kdecan_time))
    #         # print(kdecan_time)
    #         # print(kdecan_throttle)
    #
    #         ulg_thtl = ulg_thtl_dict[escid_key]
    #         ulg_time = ulg_thtl.index - ulg_thtl.index[0]
    #         ulg_time = np.array(ulg_time)
    #         ulg_throttle = np.array(ulg_thtl.values)
    #         ulg_dict[escid_key] = copy.deepcopy([ulg_time, ulg_throttle])
    #         # print(type(ulg_time))
    #         # print(ulg_time)
    #         # print(ulg_throttle)
    #
    #     return [num_esc, kdecan_dict, ulg_dict]

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

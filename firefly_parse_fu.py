import copy
import datetime
import argparse
import pandas
import numpy as np
import os

from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter

# from firefly_database import FileTagData
from firefly_parse_keys import UlgDictKeys
from firefly_parse_keys import FireflyDfKeys, FireflyDfKeysMi
from firefly_parse_keys import ArmDfKeys, ArmDfKeysMi
from firefly_parse_keys import UlgInDfKeys, UlgOutDfKeys, UlgPvDfKeys, \
    UlgAngvelDf
from toopazo_tools.pandas import DataframeTools
from toopazo_tools.file_folder import FileFolderTools as FFTools
from toopazo_ulg.parse_file import UlgParser, UlgParserTools


class FUParser:
    """
    Class to get data from log
    """

    def __init__(self, bdir, firefly_file, ulg_file):
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

        if not os.path.isfile(firefly_file):
            raise RuntimeError(f'[KUParser] No such file {firefly_file}')
        else:
            self.firefly_file = firefly_file
            self.firefly_df = FUParser.get_pandas_dataframe(firefly_file)

        if not os.path.isfile(ulg_file):
            raise RuntimeError(f'[KUParser] No such file {ulg_file}')
        else:
            self.ulg_file = ulg_file
            UlgParser.check_ulog2csv(self.tmpdir, self.ulg_file)

            ## this dict stores position and velocity data
            self.ulg_dict = UlgParser.get_ulg_dict(self.tmpdir, ulg_file)

            # Keep special dataframes separated
            self.ulg_sticks_df = self.ulg_dict['ulg_sticks_df']
            self.ulg_switches_df = self.ulg_dict['ulg_switches_df']
            # Redefine ulg_dict leaving out slowly sampled dataframes
            del self.ulg_dict['ulg_sticks_df']
            del self.ulg_dict['ulg_switches_df']

    @staticmethod
    def parse_line_escid(line, escid):
        indx = 20 + 9 * (escid - 11)
        dtformat = ' %Y-%m-%d %H:%M:%S.%f'

        try:
            escid_time = datetime.datetime.strptime(line[indx+0], dtformat)
        except ValueError:
            escid_time = float(line[indx+0])

        line_dict = {
            # 'time': datetime.datetime.strptime(line[indx+0], dtformat),
            'time': escid_time,
            'escid': int(line[indx+1]),
            'voltage': float(line[indx+2]),
            'current': float(line[indx+3]),
            'rpm': float(line[indx+4]),
            # 'temp': float(line[indx+5]),
            # 'warning': str(line[indx+6]),
            'inthrottle': float(line[indx+7]),
            'outthrottle': float(line[indx+8]),
        }
        return line_dict

    @staticmethod
    def parse_line_ars(line):
        line_dict = {
            # 'sps': int(line[0]),
            'mills': int(line[1]),
            'secs': float(line[2]),
            # 'dtmills': int(line[3]),
            'rpm1': float(line[4]),
            'rpm2': float(line[5]),
            'rpm3': float(line[6]),
            'rpm4': float(line[7]),
            'rpm5': float(line[8]),
            'rpm6': float(line[9]),
            'rpm7': float(line[10]),
            'rpm8': float(line[11]),
            'cur1': float(line[12]),
            'cur2': float(line[13]),
            'cur3': float(line[14]),
            'cur4': float(line[15]),
            'cur5': float(line[16]),
            'cur6': float(line[17]),
            'cur7': float(line[18]),
            'cur8': float(line[19]),
            # adding reordered data
            'rpm_13': float(line[4]),  # rpm1
            'rpm_18': float(line[5]),  # rpm2
            'rpm_14': float(line[6]),  # rpm3
            'rpm_17': float(line[7]),  # rpm4
            'cur_13': float(line[12]),  # cur1
            'cur_18': float(line[13]),  # cur2
            'cur_14': float(line[14]),  # cur3
            'cur_17': float(line[15]),  # cur4
        }
        return line_dict

    @staticmethod
    def parse_line_fcost(line):
        indx = 20 + 9 * 8
        # firefly_live_v4.py
        # cost = parsed_data[f'voltage_{escid}'] * parsed_data[
        #                     f'cur_{escid}']
        line_dict = {
            'fcost1': float(line[indx+0].replace('[', '')),
            'fcost2': float(line[indx+1]),
            'fcost3': float(line[indx+2]),
            'fcost4': float(line[indx+3]),
            'fcost5': float(line[indx+4]),
            'fcost6': float(line[indx+5]),
            'fcost7': float(line[indx+6]),
            'fcost8': float(line[indx+7].replace(']', '')),
        }
        return line_dict

    @staticmethod
    def parse_line_optim(line):
        indx = 20 + 9 * 8 + 8
        # firefly_live_v4.py
        # optim_data = f'{nsh_cmd}, {avg_cost_m38}, {avg_cost_m47}, ' \
        #              f'{avg_cost_tot}, {avg_cost_tot_prev}'
        line_dict = {
            'nsh_cmd': float(line[indx+0].replace('[', '')),
            'avg_cost_m38': float(line[indx+1]),
            'avg_cost_m47': float(line[indx+2]),
            'avg_cost_tot': float(line[indx+3]),
            'avg_cost_tot_prev': float(line[indx+4]),
        }
        return line_dict

    @staticmethod
    def parse_line(line):
        assert isinstance(line, list)
        llen = len(line)
        if llen == 105:
            line_dict = {}

            ars_dict = FUParser.parse_line_ars(line)
            for key in ars_dict.keys():
                line_dict[f'ars_{key}'] = ars_dict[key]

            for i in range(11, 19):
                escid = f'esc{i}'
                escid_dict = FUParser.parse_line_escid(line, escid=i)
                for key in escid_dict.keys():
                    line_dict[f'{escid}_{key}'] = escid_dict[key]

            fcost_dict = FUParser.parse_line_fcost(line)
            for key in fcost_dict.keys():
                line_dict[f'fcost_{key}'] = fcost_dict[key]

            optim_dict = FUParser.parse_line_optim(line)
            for key in optim_dict.keys():
                line_dict[f'optim_{key}'] = optim_dict[key]

            # pprint(line_dict)
            return line_dict
        else:
            return None

    @staticmethod
    def append_non_dict_values(new_dict, old_dict):
        assert isinstance(new_dict, dict)
        assert isinstance(old_dict, dict)
        # print('append_non_dict_values')
        for key in new_dict.keys():
            if isinstance(new_dict[key], dict):
                msg = f'isinstance(new_dict[{key}], dict) is True'
                raise RuntimeError(msg)
            if key not in old_dict.keys():
                old_dict[key] = [new_dict[key]]
            else:
                old_dict[key].append(copy.deepcopy(new_dict[key]))
            # print(f'[append_non_dict_values] '
            #       f'len(old_dict[{key}]) = {len(old_dict[key])}')
        return old_dict

    @staticmethod
    def get_pandas_dataframe(firefly_file):

        fd = open(firefly_file, 'r')
        cnt = 0
        firefly_dict = {}
        for line in fd.readlines():
            cnt = cnt + 1
            # print(f'cnt {cnt}, line {line}')
            line = line.strip()
            line = line.split(',')
            # num_elem = len(line)
            # print(f'cnt {cnt}, num_elem {num_elem}')
            line_dict = FUParser.parse_line(line)
            if line_dict is not None:
                firefly_dict = FUParser.append_non_dict_values(
                    copy.deepcopy(line_dict), firefly_dict)
        fd.close()

        # Check that all keys have the same length
        num_samples = []
        for key in firefly_dict.keys():
            val = firefly_dict[key]
            num_samples.append(len(val))
        # print(num_samples)
        # print(np.diff(num_samples))
        if np.sum(np.diff(num_samples)) != 0:
            raise RuntimeError('np.sum(np.diff(num_samples)) is not zero')

        # Convert to dataframe
        firefly_df = pandas.DataFrame.from_dict(
            data=firefly_dict, orient='columns', dtype=None, columns=None)
        firefly_df.index = firefly_df['esc11_time']
        return firefly_df

    @staticmethod
    def reset_index(firefly_df, ulg_dict):

        elap_time = DataframeTools.index_to_elapsed_time(firefly_df)
        firefly_df.index = elap_time
        # print(df)

        # ulg_out_df = ulg_dict[UlgDictKeys.ulg_out_df]
        # elap_time = DataframeTools.index_to_elapsed_time(ulg_out_df)
        # ulg_out_df.index = elap_time
        # print(ulg_out_df)

        for key in ulg_dict.keys():
            elap_time = DataframeTools.index_to_elapsed_time(ulg_dict[key])
            ulg_dict[key].index = elap_time
            # print(ulg_dict[key])

        return [firefly_df, ulg_dict]

    @staticmethod
    def apply_delay_and_reset_index(ulg_dict, delay):
        ulg_out_df = ulg_dict[UlgDictKeys.ulg_out_df]
        # delay = FUParser.get_ulg_delay(file_tag)
        mask = ulg_out_df.index >= delay
        for key, df in ulg_dict.items():
            df = df.loc[mask]
            df.index = df.index - df.index[0]
            ulg_dict[key] = df
        return copy.deepcopy(ulg_dict)

    @staticmethod
    def resample_firefly_df(firefly_df, new_index, verbose):
        if verbose:
            print('Inside synchronize_df_dict')
            print('Before')
            print(firefly_df)

        new_firefly_df = FUParser.resample_firefly_df_by_interp(
            firefly_df, new_index)

        if verbose:
            print('After')
            print(new_firefly_df)

        return copy.deepcopy(new_firefly_df)

    @staticmethod
    def resample_firefly_df_by_interp(firefly_df, new_index):
        x = new_index
        xp = firefly_df.index
        data = FUParser.interpolate_columns(x, xp, firefly_df)
        new_df = pandas.DataFrame(data=data, index=x)
        return copy.deepcopy(new_df)

    @staticmethod
    def interpolate_columns(x, xp, firefly_df):
        assert isinstance(firefly_df, pandas.DataFrame)
        data = {}
        black_list = [
            'esc11_time', 'esc12_time', 'esc13_time', 'esc14_time',
            'esc15_time', 'esc16_time', 'esc17_time', 'esc18_time'
        ]
        for key in firefly_df.keys():
            if key in black_list:
                continue
            key_data = np.array(firefly_df[key].values)
            data[key] = np.interp(x, xp, fp=key_data)
        return data

    @staticmethod
    def get_hover_mask(norm_arr, ulg_df_arr, arm_df):
        [vel_norm, pqr_norm] = norm_arr
        [ulg_pv_df, ulg_in_df, ulg_out_df] = ulg_df_arr

        ti = ulg_pv_df.index[0] + 10
        tf = ulg_pv_df.index[-1] - 10
        mask_norm = (ulg_pv_df.index > ti) & \
                    (ulg_pv_df.index < tf) & \
                    (vel_norm < 0.25) & \
                    (pqr_norm < 2.5) & \
                    (ulg_in_df[UlgInDfKeys.az_cmd] > 0.45) & \
                    (ulg_out_df[UlgOutDfKeys.output_0] > 1200)

        max_abs_val = 10
        mask_thr = (arm_df[ArmDfKeys.m1.rate_thr] < +max_abs_val) & \
                   (arm_df[ArmDfKeys.m1.rate_thr] > -max_abs_val) & \
                   (arm_df[ArmDfKeys.m2.rate_thr] < +max_abs_val) & \
                   (arm_df[ArmDfKeys.m2.rate_thr] > -max_abs_val) & \
                   (arm_df[ArmDfKeys.m3.rate_thr] < +max_abs_val) & \
                   (arm_df[ArmDfKeys.m3.rate_thr] > -max_abs_val) & \
                   (arm_df[ArmDfKeys.m4.rate_thr] < +max_abs_val) & \
                   (arm_df[ArmDfKeys.m4.rate_thr] > -max_abs_val) & \
                   (arm_df[ArmDfKeys.m5.rate_thr] < +max_abs_val) & \
                   (arm_df[ArmDfKeys.m5.rate_thr] > -max_abs_val) & \
                   (arm_df[ArmDfKeys.m6.rate_thr] < +max_abs_val) & \
                   (arm_df[ArmDfKeys.m6.rate_thr] > -max_abs_val) & \
                   (arm_df[ArmDfKeys.m7.rate_thr] < +max_abs_val) & \
                   (arm_df[ArmDfKeys.m7.rate_thr] > -max_abs_val) & \
                   (arm_df[ArmDfKeys.m8.rate_thr] < +max_abs_val) & \
                   (arm_df[ArmDfKeys.m8.rate_thr] > -max_abs_val)

        max_abs_val = 25
        mask_rpm = (arm_df[ArmDfKeys.m1.rate_rpm] < +max_abs_val) & \
                   (arm_df[ArmDfKeys.m1.rate_rpm] > -max_abs_val) & \
                   (arm_df[ArmDfKeys.m2.rate_rpm] < +max_abs_val) & \
                   (arm_df[ArmDfKeys.m2.rate_rpm] > -max_abs_val) & \
                   (arm_df[ArmDfKeys.m3.rate_rpm] < +max_abs_val) & \
                   (arm_df[ArmDfKeys.m3.rate_rpm] > -max_abs_val) & \
                   (arm_df[ArmDfKeys.m4.rate_rpm] < +max_abs_val) & \
                   (arm_df[ArmDfKeys.m4.rate_rpm] > -max_abs_val) & \
                   (arm_df[ArmDfKeys.m5.rate_rpm] < +max_abs_val) & \
                   (arm_df[ArmDfKeys.m5.rate_rpm] > -max_abs_val) & \
                   (arm_df[ArmDfKeys.m6.rate_rpm] < +max_abs_val) & \
                   (arm_df[ArmDfKeys.m6.rate_rpm] > -max_abs_val) & \
                   (arm_df[ArmDfKeys.m7.rate_rpm] < +max_abs_val) & \
                   (arm_df[ArmDfKeys.m7.rate_rpm] > -max_abs_val) & \
                   (arm_df[ArmDfKeys.m8.rate_rpm] < +max_abs_val) & \
                   (arm_df[ArmDfKeys.m8.rate_rpm] > -max_abs_val)

        max_abs_val = 0.5
        mask_cur = (arm_df[ArmDfKeys.m1.rate_cur] < +max_abs_val) & \
                   (arm_df[ArmDfKeys.m1.rate_cur] > -max_abs_val) & \
                   (arm_df[ArmDfKeys.m2.rate_cur] < +max_abs_val) & \
                   (arm_df[ArmDfKeys.m2.rate_cur] > -max_abs_val) & \
                   (arm_df[ArmDfKeys.m3.rate_cur] < +max_abs_val) & \
                   (arm_df[ArmDfKeys.m3.rate_cur] > -max_abs_val) & \
                   (arm_df[ArmDfKeys.m4.rate_cur] < +max_abs_val) & \
                   (arm_df[ArmDfKeys.m4.rate_cur] > -max_abs_val) & \
                   (arm_df[ArmDfKeys.m5.rate_cur] < +max_abs_val) & \
                   (arm_df[ArmDfKeys.m5.rate_cur] > -max_abs_val) & \
                   (arm_df[ArmDfKeys.m6.rate_cur] < +max_abs_val) & \
                   (arm_df[ArmDfKeys.m6.rate_cur] > -max_abs_val) & \
                   (arm_df[ArmDfKeys.m7.rate_cur] < +max_abs_val) & \
                   (arm_df[ArmDfKeys.m7.rate_cur] > -max_abs_val) & \
                   (arm_df[ArmDfKeys.m8.rate_cur] < +max_abs_val) & \
                   (arm_df[ArmDfKeys.m8.rate_cur] > -max_abs_val)

        mask = mask_norm & mask_thr & mask_rpm & mask_cur
        return mask

    @staticmethod
    def filter_by_hover_v1(firefly_df, arm_df, ulg_dict):
        minv = -30
        maxv = +30
        ref_df = arm_df
        mask = \
            (ref_df[ArmDfKeys.m3.rate_rpm] > minv) & \
            (ref_df[ArmDfKeys.m3.rate_rpm] < maxv) & \
            (ref_df[ArmDfKeys.m4.rate_rpm] > minv) & \
            (ref_df[ArmDfKeys.m4.rate_rpm] < maxv) & \
            (ref_df[ArmDfKeys.m7.rate_rpm] > minv) & \
            (ref_df[ArmDfKeys.m7.rate_rpm] < maxv) & \
            (ref_df[ArmDfKeys.m8.rate_rpm] > minv) & \
            (ref_df[ArmDfKeys.m8.rate_rpm] < maxv)
        [firefly_df, arm_df, ulg_dict] = FUParser.filter_by_mask(
            mask, firefly_df, arm_df, ulg_dict)

        ref_df = ulg_dict[UlgDictKeys.ulg_in_df]
        mask = ref_df[UlgInDfKeys.az_cmd] > 0.45
        [firefly_df, arm_df, ulg_dict] = FUParser.filter_by_mask(
            mask, firefly_df, arm_df, ulg_dict)

        ref_df = ulg_dict[UlgDictKeys.ulg_out_df]
        mask = ref_df[UlgOutDfKeys.output_0] > 1200
        [firefly_df, arm_df, ulg_dict] = FUParser.filter_by_mask(
            mask, firefly_df, arm_df, ulg_dict)

        ref_df = ulg_dict[UlgDictKeys.ulg_pv_df]
        mask = ref_df[UlgPvDfKeys.vel_norm] < 0.3
        [firefly_df, arm_df, ulg_dict] = FUParser.filter_by_mask(
            mask, firefly_df, arm_df, ulg_dict)

        ref_df = ulg_dict[UlgDictKeys.ulg_in_df]
        mask = (ref_df[UlgInDfKeys.r_rate_cmd] > -0.04) & (
                ref_df[UlgInDfKeys.r_rate_cmd] < +0.04)
        [firefly_df, arm_df, ulg_dict] = FUParser.filter_by_mask(
            mask, firefly_df, arm_df, ulg_dict)

        return [firefly_df, arm_df, ulg_dict]

    @staticmethod
    def filter_by_hover_v2(firefly_df, arm_df, ulg_dict):
        ulg_in_df = ulg_dict[UlgDictKeys.ulg_in_df]
        ulg_out_df = ulg_dict[UlgDictKeys.ulg_out_df]
        ulg_pv_df = ulg_dict[UlgDictKeys.ulg_pv_df]
        ulg_angvel_df = ulg_dict[UlgDictKeys.ulg_angvel_df]
        # ulg_accel_df = ulg_dict[UlgDictKeys.ulg_accel_df]
        # ulg_angacc_df = ulg_dict[UlgDictKeys.ulg_angacc_df]

        # t_outgnd = fu_data_dict['t_outgnd']
        # time0 = t_outgnd[0]
        # time1 = t_outgnd[1]
        # mask = (firefly_df.index > time0) & (firefly_df.index <= time1)

        # ti = ulg_pv_df.index[0] + 0
        # tf = ulg_pv_df.index[-1] - 10
        # mask = (ulg_pv_df.index > ti) & \
        #        (ulg_pv_df.index < tf) & \
        mask = (ulg_pv_df[UlgPvDfKeys.vel_norm] < 0.25) & \
               (ulg_angvel_df[UlgAngvelDf.pqr_norm] < 2.5) & \
               (ulg_in_df[UlgInDfKeys.az_cmd] > 0.45) & \
               (ulg_out_df[UlgOutDfKeys.output_0] > 1200)
        [firefly_df, arm_df, ulg_dict] = FUParser.filter_by_mask(
            mask, firefly_df, arm_df, ulg_dict)

        return [firefly_df, arm_df, ulg_dict]

    @staticmethod
    def filter_by_hover_v3(firefly_df, arm_df, ulg_dict):
        ulg_in_df = ulg_dict[UlgDictKeys.ulg_in_df]
        ulg_out_df = ulg_dict[UlgDictKeys.ulg_out_df]
        ulg_pv_df = ulg_dict[UlgDictKeys.ulg_pv_df]
        ulg_angvel_df = ulg_dict[UlgDictKeys.ulg_angvel_df]
        # ulg_angacc_df = ulg_dict[UlgDictKeys.ulg_angacc_df]
        # ulg_accel_df = ulg_dict[UlgDictKeys.ulg_accel_df]
        vel_norm = ulg_pv_df[UlgPvDfKeys.vel_norm]
        pqr_norm = ulg_angvel_df[UlgAngvelDf.pqr_norm]
        # acc_norm = ulg_accel_df[UlgAccelDf.acc_norm]
        # angacc_norm = ulg_angacc_df[UlgAngaccDf.angacc_norm]

        mask = FUParser.get_hover_mask(
            norm_arr=[vel_norm, pqr_norm],
            ulg_df_arr=[ulg_pv_df, ulg_in_df, ulg_out_df],
            arm_df=arm_df)

        [firefly_df, arm_df, ulg_dict] = FUParser.filter_by_mask(
            mask, firefly_df, arm_df, ulg_dict)

        return [firefly_df, arm_df, ulg_dict]

    @staticmethod
    def filter_by_mask(mask, firefly_df, arm_df, ulg_dict):
        mask.index = firefly_df.index
        firefly_df = firefly_df[mask]

        mask.index = arm_df.index
        arm_df = arm_df[mask]

        ulg_dict = UlgParserTools.remove_by_condition(ulg_dict, mask)
        return [firefly_df, arm_df, ulg_dict]

    @staticmethod
    def check_current(firefly_df):
        ars_cur_13 = firefly_df[FireflyDfKeys.m3.cur_ars].values
        ars_cur_18 = firefly_df[FireflyDfKeys.m8.cur_ars].values
        ars_cur_14 = firefly_df[FireflyDfKeys.m4.cur_ars].values
        ars_cur_17 = firefly_df[FireflyDfKeys.m7.cur_ars].values

        esc_cur_13 = firefly_df[FireflyDfKeys.m3.cur].values
        esc_cur_18 = firefly_df[FireflyDfKeys.m8.cur].values
        esc_cur_14 = firefly_df[FireflyDfKeys.m4.cur].values
        esc_cur_17 = firefly_df[FireflyDfKeys.m7.cur].values

        # reset_current
        num_samples = 10

        print('ars current')
        data_arr = [ars_cur_13, ars_cur_18, ars_cur_14, ars_cur_17]
        mi_arr = [3, 8, 4, 7]
        for mi, data in zip(mi_arr, data_arr):
            val1 = np.mean(data[:num_samples])
            print('m%s, avg [:%s] %s' % (mi, num_samples, val1))
            val2 = np.std(data)
            print('m%s, std %s' % (mi, val2))

        print('esc current')
        data_arr = [esc_cur_13, esc_cur_18, esc_cur_14, esc_cur_17]
        mi_arr = [3, 8, 4, 7]
        for mi, data in zip(mi_arr, data_arr):
            val1 = np.mean(data[:num_samples])
            print('m%s, avg [:%s] %s' % (mi, num_samples, val1))
            val2 = np.std(data)
            print('m%s, std %s' % (mi, val2))

    @staticmethod
    def smooth_current(firefly_df):
        iarr = [1, 2, 3, 4, 5, 6, 7, 8]
        # iarr = [3, 8]
        # iarr = [4, 7]
        # window size 51, polynomial order 3
        for i in iarr:
            ff_key_mi = FireflyDfKeysMi(i)
            y = firefly_df[ff_key_mi.cur]
            yhat = savgol_filter(y, 51, 3)
            firefly_df[ff_key_mi.cur] = yhat
            try:
                y = firefly_df[ff_key_mi.cur_ars]
                yhat = savgol_filter(y, 51, 3)
                firefly_df[ff_key_mi.cur_ars] = yhat
            except KeyError:
                pass

        return firefly_df

    @staticmethod
    def calibrate_current(firefly_df, file_tag, get_cur_0rpm):
        ff_df = copy.deepcopy(firefly_df)

        # Measuring Current Using ACS712
        # int mVperAmp = 185; // use 100 for 20A Module and 66 for 30A Module
        mv_per_amp = 100
        # int ACSoffset = 2500;
        acs_offset = 2500
        # RawValue = analogRead(analogIn);
        # Voltage = (RawValue / 1024.0) * 5000; // Gets you mV
        # Amps = ((Voltage - ACSoffset) / mVperAmp);

        manual_ars_offset = 1.7 * 1

        key = FireflyDfKeys.m3.cur_ars
        raw_value = ff_df[key]
        milli_volts = (raw_value / 1024.0) * 5000
        amps = (milli_volts - acs_offset) / mv_per_amp
        ff_df[key] = amps - manual_ars_offset

        key = FireflyDfKeys.m4.cur_ars
        raw_value = ff_df[key]
        milli_volts = (raw_value / 1024.0) * 5000
        amps = (milli_volts - acs_offset) / mv_per_amp
        ff_df[key] = amps - manual_ars_offset

        key = FireflyDfKeys.m7.cur_ars
        raw_value = ff_df[key]
        milli_volts = (raw_value / 1024.0) * 5000
        amps = (milli_volts - acs_offset) / mv_per_amp
        ff_df[key] = amps - manual_ars_offset

        key = FireflyDfKeys.m8.cur_ars
        raw_value = ff_df[key]
        milli_volts = (raw_value / 1024.0) * 5000
        amps = (milli_volts - acs_offset) / mv_per_amp
        ff_df[key] = amps - manual_ars_offset

        if get_cur_0rpm:
            cur_0rpm_arr = []
            for mi in range(1, 9):
                ff_key_mi = FireflyDfKeysMi(mi)
                mask = (ff_df[ff_key_mi.thr] < 950) & \
                       (ff_df[ff_key_mi.rpm] < 10)
                # ff_0rpm_df = ff_df[mask][
                #     [ff_key_mi.thr, ff_key_mi.rpm, ff_key_mi.cur]
                # ]
                # print(ff_0rpm_df)
                val_arr = ff_df[mask][ff_key_mi.cur].values
                if len(val_arr) > 5:
                    cur_0rpm = np.mean(val_arr)
                    # print(f'Zero rpm current is {cur_0rpm} for m{mi}')
                else:
                    cur_0rpm = np.nan
                    # print(f'Not enough data to calculate cur_0rpm for m{mi}')
                cur_0rpm_arr.append(round(cur_0rpm, 2))

                ff_df[ff_key_mi.cur] = ff_df[ff_key_mi.cur] - cur_0rpm
                # mask = pandas.Series(mask)

            print(f'cur_0rpm_arr {cur_0rpm_arr}')
            return cur_0rpm_arr

        # esc_cal_type = 'cur_0rpm_arr'
        # esc_cal_type = 'ars_min_cur_static'
        esc_cal_type = 'ars_min_cur_outgnd'
        # esc_cal_type = 'use_file_tag'
        if esc_cal_type == 'cur_0rpm_arr':
            m1_c0rpm = 6.611666666666667
            m2_c0rpm = 2.3453846153846154
            m3_c0rpm = 0.0
            m4_c0rpm = 0.5561538461538462
            m5_c0rpm = 6.481538461538462
            m6_c0rpm = 3.174615384615385
            m7_c0rpm = 0.0
            m8_c0rpm = 0.0015384615384615385
            ff_df[FireflyDfKeys.m1.cur] = ff_df[FireflyDfKeys.m1.cur] - m1_c0rpm
            ff_df[FireflyDfKeys.m2.cur] = ff_df[FireflyDfKeys.m2.cur] - m2_c0rpm
            ff_df[FireflyDfKeys.m3.cur] = ff_df[FireflyDfKeys.m3.cur] - m3_c0rpm
            ff_df[FireflyDfKeys.m4.cur] = ff_df[FireflyDfKeys.m4.cur] - m4_c0rpm
            ff_df[FireflyDfKeys.m5.cur] = ff_df[FireflyDfKeys.m5.cur] - m5_c0rpm
            ff_df[FireflyDfKeys.m6.cur] = ff_df[FireflyDfKeys.m6.cur] - m6_c0rpm
            ff_df[FireflyDfKeys.m7.cur] = ff_df[FireflyDfKeys.m7.cur] - m7_c0rpm
            ff_df[FireflyDfKeys.m8.cur] = ff_df[FireflyDfKeys.m8.cur] - m8_c0rpm

        if esc_cal_type == 'ars_min_cur_static':
            key = FireflyDfKeys.m1.cur
            delta_upper = 2.5 - np.min(ff_df[key].values)
            ff_df[key] = ff_df[key] + delta_upper
            key = FireflyDfKeys.m2.cur
            delta_upper = 2.5 - np.min(ff_df[key].values)
            ff_df[key] = ff_df[key] + delta_upper
            key = FireflyDfKeys.m3.cur
            delta_upper = 2.5 - np.min(ff_df[key].values)
            ff_df[key] = ff_df[key] + delta_upper
            key = FireflyDfKeys.m4.cur
            delta_upper = 2.5 - np.min(ff_df[key].values)
            ff_df[key] = ff_df[key] + delta_upper

            key = FireflyDfKeys.m5.cur
            delta_lower = 5.2 - np.min(ff_df[key].values)
            ff_df[key] = ff_df[key] + delta_lower
            key = FireflyDfKeys.m6.cur
            delta_lower = 5.2 - np.min(ff_df[key].values)
            ff_df[key] = ff_df[key] + delta_lower
            key = FireflyDfKeys.m7.cur
            delta_lower = 5.2 - np.min(ff_df[key].values)
            ff_df[key] = ff_df[key] + delta_lower
            key = FireflyDfKeys.m8.cur
            delta_lower = 5.2 - np.min(ff_df[key].values)
            ff_df[key] = ff_df[key] + delta_lower

        if esc_cal_type == 'ars_min_cur_outgnd':
            for i in [1, 2, 3, 4]:
                ff_key_mi = FireflyDfKeysMi(i)
                m3_min_ars = np.min(ff_df[FireflyDfKeys.m3.cur_ars].values)
                m4_min_ars = np.min(ff_df[FireflyDfKeys.m4.cur_ars].values)
                min_ars = 0.5 * (m3_min_ars + m4_min_ars)
                min_esc = np.min(ff_df[ff_key_mi.cur].values)
                delta_upper = min_ars - min_esc
                ff_df[ff_key_mi.cur] = ff_df[ff_key_mi.cur] + delta_upper
            for i in [5, 6, 7, 8]:
                ff_key_mi = FireflyDfKeysMi(i)
                m7_min_ars = np.min(ff_df[FireflyDfKeys.m7.cur_ars].values)
                m8_min_ars = np.min(ff_df[FireflyDfKeys.m8.cur_ars].values)
                min_ars = 0.5 * (m7_min_ars + m8_min_ars)
                min_esc = np.min(ff_df[ff_key_mi.cur].values)
                delta_upper = min_ars - min_esc
                ff_df[ff_key_mi.cur] = ff_df[ff_key_mi.cur] + delta_upper

        # if esc_cal_type == 'use_file_tag':
        #     data_dict = FileTagData.data_dict(file_tag)
        #     cbias = data_dict['cur_bias']
        #     ff_df[FireflyDfKeys.m1.cur] = ff_df[FireflyDfKeys.m1.cur] + cbias[0]
        #     ff_df[FireflyDfKeys.m2.cur] = ff_df[FireflyDfKeys.m2.cur] + cbias[1]
        #     ff_df[FireflyDfKeys.m3.cur] = ff_df[FireflyDfKeys.m3.cur] + cbias[2]
        #     ff_df[FireflyDfKeys.m4.cur] = ff_df[FireflyDfKeys.m4.cur] + cbias[3]
        #     ff_df[FireflyDfKeys.m5.cur] = ff_df[FireflyDfKeys.m5.cur] + cbias[4]
        #     ff_df[FireflyDfKeys.m6.cur] = ff_df[FireflyDfKeys.m6.cur] + cbias[5]
        #     ff_df[FireflyDfKeys.m7.cur] = ff_df[FireflyDfKeys.m7.cur] + cbias[6]
        #     ff_df[FireflyDfKeys.m8.cur] = ff_df[FireflyDfKeys.m8.cur] + cbias[7]

        smooth = False
        if smooth:
            for mi in range(1, 9):
                ff_df[FireflyDfKeysMi(mi).cur] = pandas.DataFrame(
                    index=ff_df.index, data={'data': savgol_filter(
                        ff_df[FireflyDfKeysMi(mi).cur].values, 21, 3)}
                )['data']

        return ff_df

    @staticmethod
    def get_arm_df(firefly_df):
        ff_df = firefly_df
        arm_df = copy.deepcopy(ff_df)
        arm_df.drop(columns=arm_df.keys(), inplace=True)

        ff_key_m1 = FireflyDfKeys.m1
        ff_key_m2 = FireflyDfKeys.m2
        ff_key_m3 = FireflyDfKeys.m3
        ff_key_m4 = FireflyDfKeys.m4
        ff_key_m5 = FireflyDfKeys.m5
        ff_key_m6 = FireflyDfKeys.m6
        ff_key_m7 = FireflyDfKeys.m7
        ff_key_m8 = FireflyDfKeys.m8

        ArmDfKeys.m1 = ArmDfKeys.m1
        ArmDfKeys.m2 = ArmDfKeys.m2
        ArmDfKeys.m3 = ArmDfKeys.m3
        ArmDfKeys.m4 = ArmDfKeys.m4
        ArmDfKeys.m5 = ArmDfKeys.m5
        ArmDfKeys.m6 = ArmDfKeys.m6
        ArmDfKeys.m7 = ArmDfKeys.m7
        ArmDfKeys.m8 = ArmDfKeys.m8

        # Ratios (thr, rpm, cur)
        m16_eta_thr = ff_df[ff_key_m1.thr] / ff_df[ff_key_m6.thr]
        m25_eta_thr = ff_df[ff_key_m2.thr] / ff_df[ff_key_m5.thr]
        m38_eta_thr = ff_df[ff_key_m3.thr] / ff_df[ff_key_m8.thr]
        m47_eta_thr = ff_df[ff_key_m4.thr] / ff_df[ff_key_m7.thr]
        arm_df[ArmDfKeys.m16.eta_thr] = m16_eta_thr.values
        arm_df[ArmDfKeys.m25.eta_thr] = m25_eta_thr.values
        arm_df[ArmDfKeys.m38.eta_thr] = m38_eta_thr.values
        arm_df[ArmDfKeys.m47.eta_thr] = m47_eta_thr.values

        m16_eta_rpm = ff_df[ff_key_m1.rpm] / ff_df[ff_key_m6.rpm]
        m25_eta_rpm = ff_df[ff_key_m2.rpm] / ff_df[ff_key_m5.rpm]
        m38_eta_rpm = ff_df[ff_key_m3.rpm] / ff_df[ff_key_m8.rpm]
        m47_eta_rpm = ff_df[ff_key_m4.rpm] / ff_df[ff_key_m7.rpm]
        arm_df[ArmDfKeys.m16.eta_rpm] = m16_eta_rpm.values
        arm_df[ArmDfKeys.m25.eta_rpm] = m25_eta_rpm.values
        arm_df[ArmDfKeys.m38.eta_rpm] = m38_eta_rpm.values
        arm_df[ArmDfKeys.m47.eta_rpm] = m47_eta_rpm.values

        m16_eta_cur = ff_df[ff_key_m1.cur] / ff_df[ff_key_m6.cur]
        m25_eta_cur = ff_df[ff_key_m2.cur] / ff_df[ff_key_m5.cur]
        m38_eta_cur = ff_df[ff_key_m3.cur] / ff_df[ff_key_m8.cur]
        m47_eta_cur = ff_df[ff_key_m4.cur] / ff_df[ff_key_m7.cur]
        arm_df[ArmDfKeys.m16.eta_cur] = m16_eta_cur.values
        arm_df[ArmDfKeys.m25.eta_cur] = m25_eta_cur.values
        arm_df[ArmDfKeys.m38.eta_cur] = m38_eta_cur.values
        arm_df[ArmDfKeys.m47.eta_cur] = m47_eta_cur.values

        # Deltas (thr, rpm, cur)
        m16_delta_thr = ff_df[ff_key_m1.thr] - ff_df[ff_key_m6.thr]
        m25_delta_thr = ff_df[ff_key_m2.thr] - ff_df[ff_key_m5.thr]
        m38_delta_thr = ff_df[ff_key_m3.thr] - ff_df[ff_key_m8.thr]
        m47_delta_thr = ff_df[ff_key_m4.thr] - ff_df[ff_key_m7.thr]
        arm_df[ArmDfKeys.m16.delta_thr] = m16_delta_thr.values
        arm_df[ArmDfKeys.m25.delta_thr] = m25_delta_thr.values
        arm_df[ArmDfKeys.m38.delta_thr] = m38_delta_thr.values
        arm_df[ArmDfKeys.m47.delta_thr] = m47_delta_thr.values

        m16_delta_rpm = ff_df[ff_key_m1.rpm] - ff_df[ff_key_m6.rpm]
        m25_delta_rpm = ff_df[ff_key_m2.rpm] - ff_df[ff_key_m5.rpm]
        m38_delta_rpm = ff_df[ff_key_m3.rpm] - ff_df[ff_key_m8.rpm]
        m47_delta_rpm = ff_df[ff_key_m4.rpm] - ff_df[ff_key_m7.rpm]
        arm_df[ArmDfKeys.m16.delta_rpm] = m16_delta_rpm.values
        arm_df[ArmDfKeys.m25.delta_rpm] = m25_delta_rpm.values
        arm_df[ArmDfKeys.m38.delta_rpm] = m38_delta_rpm.values
        arm_df[ArmDfKeys.m47.delta_rpm] = m47_delta_rpm.values

        m16_delta_cur = ff_df[ff_key_m1.cur] - ff_df[ff_key_m6.cur]
        m25_delta_cur = ff_df[ff_key_m2.cur] - ff_df[ff_key_m5.cur]
        m38_delta_cur = ff_df[ff_key_m3.cur] - ff_df[ff_key_m8.cur]
        m47_delta_cur = ff_df[ff_key_m4.cur] - ff_df[ff_key_m7.cur]
        arm_df[ArmDfKeys.m16.delta_cur] = m16_delta_cur.values
        arm_df[ArmDfKeys.m25.delta_cur] = m25_delta_cur.values
        arm_df[ArmDfKeys.m38.delta_cur] = m38_delta_cur.values
        arm_df[ArmDfKeys.m47.delta_cur] = m47_delta_cur.values

        # Rates for (thr, rpm, cur) -> d(Throttle)/dt
        m1_rate_thr = np.insert(
            np.diff(ff_df[ff_key_m1.thr].values), obj=0, values=0)
        m2_rate_thr = np.insert(
            np.diff(ff_df[ff_key_m2.thr].values), obj=0, values=0)
        m3_rate_thr = np.insert(
            np.diff(ff_df[ff_key_m3.thr].values), obj=0, values=0)
        m4_rate_thr = np.insert(
            np.diff(ff_df[ff_key_m4.thr].values), obj=0, values=0)
        m5_rate_thr = np.insert(
            np.diff(ff_df[ff_key_m5.thr].values), obj=0, values=0)
        m6_rate_thr = np.insert(
            np.diff(ff_df[ff_key_m6.thr].values), obj=0, values=0)
        m7_rate_thr = np.insert(
            np.diff(ff_df[ff_key_m7.thr].values), obj=0, values=0)
        m8_rate_thr = np.insert(
            np.diff(ff_df[ff_key_m8.thr].values), obj=0, values=0)
        arm_df[ArmDfKeys.m1.rate_thr] = m1_rate_thr
        arm_df[ArmDfKeys.m2.rate_thr] = m2_rate_thr
        arm_df[ArmDfKeys.m3.rate_thr] = m3_rate_thr
        arm_df[ArmDfKeys.m4.rate_thr] = m4_rate_thr
        arm_df[ArmDfKeys.m5.rate_thr] = m5_rate_thr
        arm_df[ArmDfKeys.m6.rate_thr] = m6_rate_thr
        arm_df[ArmDfKeys.m7.rate_thr] = m7_rate_thr
        arm_df[ArmDfKeys.m8.rate_thr] = m8_rate_thr

        # delta RPM -> d(RPM)/dt
        m1_rate_rpm = np.insert(
            np.diff(ff_df[ff_key_m1.rpm].values), obj=0, values=0)
        m2_rate_rpm = np.insert(
            np.diff(ff_df[ff_key_m2.rpm].values), obj=0, values=0)
        m3_rate_rpm = np.insert(
            np.diff(ff_df[ff_key_m3.rpm].values), obj=0, values=0)
        m4_rate_rpm = np.insert(
            np.diff(ff_df[ff_key_m4.rpm].values), obj=0, values=0)
        m5_rate_rpm = np.insert(
            np.diff(ff_df[ff_key_m5.rpm].values), obj=0, values=0)
        m6_rate_rpm = np.insert(
            np.diff(ff_df[ff_key_m6.rpm].values), obj=0, values=0)
        m7_rate_rpm = np.insert(
            np.diff(ff_df[ff_key_m7.rpm].values), obj=0, values=0)
        m8_rate_rpm = np.insert(
            np.diff(ff_df[ff_key_m8.rpm].values), obj=0, values=0)
        arm_df[ArmDfKeys.m1.rate_rpm] = m1_rate_rpm
        arm_df[ArmDfKeys.m2.rate_rpm] = m2_rate_rpm
        arm_df[ArmDfKeys.m3.rate_rpm] = m3_rate_rpm
        arm_df[ArmDfKeys.m4.rate_rpm] = m4_rate_rpm
        arm_df[ArmDfKeys.m5.rate_rpm] = m5_rate_rpm
        arm_df[ArmDfKeys.m6.rate_rpm] = m6_rate_rpm
        arm_df[ArmDfKeys.m7.rate_rpm] = m7_rate_rpm
        arm_df[ArmDfKeys.m8.rate_rpm] = m8_rate_rpm

        # -> d(current)/dt
        m1_rate_cur = np.insert(
            np.diff(ff_df[ff_key_m1.cur].values), obj=0, values=0)
        m2_rate_cur = np.insert(
            np.diff(ff_df[ff_key_m2.cur].values), obj=0, values=0)
        m3_rate_cur = np.insert(
            np.diff(ff_df[ff_key_m3.cur].values), obj=0, values=0)
        m4_rate_cur = np.insert(
            np.diff(ff_df[ff_key_m4.cur].values), obj=0, values=0)
        m5_rate_cur = np.insert(
            np.diff(ff_df[ff_key_m5.cur].values), obj=0, values=0)
        m6_rate_cur = np.insert(
            np.diff(ff_df[ff_key_m6.cur].values), obj=0, values=0)
        m7_rate_cur = np.insert(
            np.diff(ff_df[ff_key_m7.cur].values), obj=0, values=0)
        m8_rate_cur = np.insert(
            np.diff(ff_df[ff_key_m8.cur].values), obj=0, values=0)
        arm_df[ArmDfKeys.m1.rate_cur] = m1_rate_cur
        arm_df[ArmDfKeys.m2.rate_cur] = m2_rate_cur
        arm_df[ArmDfKeys.m3.rate_cur] = m3_rate_cur
        arm_df[ArmDfKeys.m4.rate_cur] = m4_rate_cur
        arm_df[ArmDfKeys.m5.rate_cur] = m5_rate_cur
        arm_df[ArmDfKeys.m6.rate_cur] = m6_rate_cur
        arm_df[ArmDfKeys.m7.rate_cur] = m7_rate_cur
        arm_df[ArmDfKeys.m8.rate_cur] = m8_rate_cur

        # Rates for (thr, rpm, cur) arms
        m16_rate_thr = m1_rate_thr + m6_rate_thr
        m25_rate_thr = m2_rate_thr + m5_rate_thr
        m38_rate_thr = m3_rate_thr + m8_rate_thr
        m47_rate_thr = m4_rate_thr + m7_rate_thr
        arm_df[ArmDfKeys.m16.rate_thr] = m16_rate_thr
        arm_df[ArmDfKeys.m25.rate_thr] = m25_rate_thr
        arm_df[ArmDfKeys.m38.rate_thr] = m38_rate_thr
        arm_df[ArmDfKeys.m47.rate_thr] = m47_rate_thr

        m16_rate_rpm = m1_rate_rpm + m6_rate_rpm
        m25_rate_rpm = m2_rate_rpm + m5_rate_rpm
        m38_rate_rpm = m3_rate_rpm + m8_rate_rpm
        m47_rate_rpm = m4_rate_rpm + m7_rate_rpm
        arm_df[ArmDfKeys.m16.rate_rpm] = m16_rate_rpm
        arm_df[ArmDfKeys.m25.rate_rpm] = m25_rate_rpm
        arm_df[ArmDfKeys.m38.rate_rpm] = m38_rate_rpm
        arm_df[ArmDfKeys.m47.rate_rpm] = m47_rate_rpm

        m16_rate_cur = m1_rate_cur + m6_rate_cur
        m25_rate_cur = m2_rate_cur + m5_rate_cur
        m38_rate_cur = m3_rate_cur + m8_rate_cur
        m47_rate_cur = m4_rate_cur + m7_rate_cur
        arm_df[ArmDfKeys.m16.rate_cur] = m16_rate_cur
        arm_df[ArmDfKeys.m25.rate_cur] = m25_rate_cur
        arm_df[ArmDfKeys.m38.rate_cur] = m38_rate_cur
        arm_df[ArmDfKeys.m47.rate_cur] = m47_rate_cur

        # Power from esc
        m1_pow_esc = ff_df[ff_key_m1.vol] * ff_df[ff_key_m1.cur]
        m2_pow_esc = ff_df[ff_key_m2.vol] * ff_df[ff_key_m2.cur]
        m3_pow_esc = ff_df[ff_key_m3.vol] * ff_df[ff_key_m3.cur]
        m4_pow_esc = ff_df[ff_key_m4.vol] * ff_df[ff_key_m4.cur]
        m5_pow_esc = ff_df[ff_key_m5.vol] * ff_df[ff_key_m5.cur]
        m6_pow_esc = ff_df[ff_key_m6.vol] * ff_df[ff_key_m6.cur]
        m7_pow_esc = ff_df[ff_key_m7.vol] * ff_df[ff_key_m7.cur]
        m8_pow_esc = ff_df[ff_key_m8.vol] * ff_df[ff_key_m8.cur]
        arm_df[ArmDfKeys.m1.pow_esc] = m1_pow_esc.values
        arm_df[ArmDfKeys.m2.pow_esc] = m2_pow_esc.values
        arm_df[ArmDfKeys.m3.pow_esc] = m3_pow_esc.values
        arm_df[ArmDfKeys.m4.pow_esc] = m4_pow_esc.values
        arm_df[ArmDfKeys.m5.pow_esc] = m5_pow_esc.values
        arm_df[ArmDfKeys.m6.pow_esc] = m6_pow_esc.values
        arm_df[ArmDfKeys.m7.pow_esc] = m7_pow_esc.values
        arm_df[ArmDfKeys.m8.pow_esc] = m8_pow_esc.values

        m16_pow_esc = m1_pow_esc + m6_pow_esc
        m25_pow_esc = m2_pow_esc + m5_pow_esc
        m38_pow_esc = m3_pow_esc + m8_pow_esc
        m47_pow_esc = m4_pow_esc + m7_pow_esc
        arm_df[ArmDfKeys.m16.pow_esc] = m16_pow_esc.values
        arm_df[ArmDfKeys.m25.pow_esc] = m25_pow_esc.values
        arm_df[ArmDfKeys.m38.pow_esc] = m38_pow_esc.values
        arm_df[ArmDfKeys.m47.pow_esc] = m47_pow_esc.values

        # Power from ars
        m1_pow_ars = ff_df[ff_key_m1.vol] * 0     # ff_df['ars_cur_11']
        m2_pow_ars = ff_df[ff_key_m2.vol] * 0     # ff_df['ars_cur_12']
        m3_pow_ars = ff_df[ff_key_m3.vol] * ff_df[ff_key_m3.cur_ars]
        m4_pow_ars = ff_df[ff_key_m4.vol] * ff_df[ff_key_m4.cur_ars]
        m5_pow_ars = ff_df[ff_key_m5.vol] * 0     # ff_df['ars_cur_15']
        m6_pow_ars = ff_df[ff_key_m6.vol] * 0     # ff_df['ars_cur_16']
        m7_pow_ars = ff_df[ff_key_m7.vol] * ff_df[ff_key_m7.cur_ars]
        m8_pow_ars = ff_df[ff_key_m8.vol] * ff_df[ff_key_m8.cur_ars]
        arm_df[ArmDfKeys.m1.pow_ars] = m1_pow_ars.values
        arm_df[ArmDfKeys.m2.pow_ars] = m2_pow_ars.values
        arm_df[ArmDfKeys.m3.pow_ars] = m3_pow_ars.values
        arm_df[ArmDfKeys.m4.pow_ars] = m4_pow_ars.values
        arm_df[ArmDfKeys.m5.pow_ars] = m5_pow_ars.values
        arm_df[ArmDfKeys.m6.pow_ars] = m6_pow_ars.values
        arm_df[ArmDfKeys.m7.pow_ars] = m7_pow_ars.values
        arm_df[ArmDfKeys.m8.pow_ars] = m8_pow_ars.values

        m16_pow_ars = m1_pow_ars + m6_pow_ars
        m25_pow_ars = m2_pow_ars + m5_pow_ars
        m38_pow_ars = m3_pow_ars + m8_pow_ars
        m47_pow_ars = m4_pow_ars + m7_pow_ars
        arm_df[ArmDfKeys.m16.pow_ars] = m16_pow_ars.values
        arm_df[ArmDfKeys.m25.pow_ars] = m25_pow_ars.values
        arm_df[ArmDfKeys.m38.pow_ars] = m38_pow_ars.values
        arm_df[ArmDfKeys.m47.pow_ars] = m47_pow_ars.values

        return arm_df

    @staticmethod
    def get_power_estimation(firefly_df, arm_df, ulg_dict):
        # x = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        # # y = 1 * x_0 + 2 * x_1 + 3
        # y = np.dot(x, np.array([1, 2])) + 3
        # lreg = LinearRegression()
        # lrfit = lreg.fit(x, y)
        # print(lrfit.score(x, y))
        # print(lrfit.coef_)
        # print(lrfit.intercept_)
        # print(lrfit.predict(np.array([[3, 5]])))

        # x.shape = 4 x 2 matrix = nsamples x nvariables
        # print(x.shape)

        powest_df = pandas.DataFrame(index=arm_df.index)

        # print("""
        #         Note about method get_power_estimation
        #         I should rather estimate power (linear regression on a time
        #         window of every flight where eta_rpm is close to 1)
        #     """)

        # https://scikit-learn.org/stable/modules/generated/
        # sklearn.linear_model.LinearRegression.html
        # linreg = LinearRegression()
        # y_hat = lreg_a * x + lreg_b

        ulg_in_df = ulg_dict[UlgDictKeys.ulg_in_df]
        p_rate_cmd = ulg_in_df[UlgInDfKeys.p_rate_cmd].values
        q_rate_cmd = ulg_in_df[UlgInDfKeys.q_rate_cmd].values
        r_rate_cmd = ulg_in_df[UlgInDfKeys.r_rate_cmd].values
        az_cmd = ulg_in_df[UlgInDfKeys.az_cmd].values
        x_mixer_in = np.array([p_rate_cmd, q_rate_cmd, r_rate_cmd, az_cmd])
        x_mixer_in = np.transpose(x_mixer_in)

        linreg_dict = FUParser.linreg_powest_v1(arm_df, firefly_df, ulg_dict)
        for mi in ['16', '25', '38', '47']:
            mi_dict = linreg_dict[f'm{mi}_linreg']
            linreg_fit = mi_dict['linreg_fit']
            # linreg_a = mi_dict['linreg_a']
            # linreg_b = mi_dict['linreg_b']
            # print(f'm{mi}')
            # print(f'linreg_fit {linreg_fit}')
            # print(f'linreg_a {linreg_a}')
            # print(f'linreg_b {linreg_b}')

            x_nsxnv = x_mixer_in
            try:
                power_est = linreg_fit.predict(np.array(x_nsxnv))
            except ValueError:
                power_est = x_nsxnv[:, 0] * 0
            # yhat_power = np.dot(x_nsxnv, lreg_a[f'm{mi}_lreg_a']) + lreg_b[
            #     f'm{mi}_lreg_b']

            powest_df[f'm{mi}_pow_est1'] = power_est
            arm_key_mi = ArmDfKeysMi(mi)
            power_esc = arm_df[arm_key_mi.pow_esc]
            powest_df[f'm{mi}_res_est1'] = power_est - power_esc

        linreg_dict = FUParser.linreg_powest_v2(arm_df, firefly_df, ulg_dict)
        for mi in ['16', '25', '38', '47']:
            ff_key_mu = FireflyDfKeysMi(mi[0])
            ff_key_ml = FireflyDfKeysMi(mi[1])
            upper_rpm = firefly_df[ff_key_mu.rpm].values
            lower_rpm = firefly_df[ff_key_ml.rpm].values
            x_rpm = np.array([upper_rpm, lower_rpm])
            x_rpm = np.transpose(x_rpm)

            mi_dict = linreg_dict[f'm{mi}_linreg']
            linreg_fit = mi_dict['linreg_fit']
            # linreg_a = mi_dict['linreg_a']
            # linreg_b = mi_dict['linreg_b']
            # print(f'm{mi}')
            # print(f'linreg_fit {linreg_fit}')
            # print(f'linreg_a {linreg_a}')
            # print(f'linreg_b {linreg_b}')

            x_nsxnv = x_rpm
            try:
                power_est = linreg_fit.predict(np.array(x_nsxnv))
            except ValueError:
                power_est = x_nsxnv[:, 0] * 0
            # yhat_power = np.dot(x_nsxnv, lreg_a[f'm{mi}_lreg_a']) + lreg_b[
            #     f'm{mi}_lreg_b']

            powest_df[f'm{mi}_pow_est2'] = power_est
            arm_key_mi = ArmDfKeysMi(mi)
            power_esc = arm_df[arm_key_mi.pow_esc]
            powest_df[f'm{mi}_res_est2'] = power_est - power_esc

        return powest_df

    @staticmethod
    def linreg_powest_v1(arm_df, firefly_df, ulg_dict):
        firefly_df = copy.deepcopy(firefly_df)
        arm_df = copy.deepcopy(arm_df)
        ulg_dict = copy.deepcopy(ulg_dict)

        # https://scikit-learn.org/stable/modules/generated/
        # sklearn.linear_model.LinearRegression.html
        linreg = LinearRegression()
        # y_hat = lreg_a * x + lreg_b

        linreg_dict = {}
        for mi in ['16', '25', '38', '47']:
            arm_key_mi = ArmDfKeysMi(mi)
            mask = (arm_df[arm_key_mi.eta_rpm] < 1.1) & (
                    arm_df[arm_key_mi.eta_rpm] > 0.9)
            mask = pandas.Series(mask)
            mask.index = firefly_df.index
            [firefly_df, arm_df, ulg_dict] = FUParser.filter_by_mask(
                mask, firefly_df, arm_df, ulg_dict)

            ulg_in_df = ulg_dict[UlgDictKeys.ulg_in_df]
            p_rate_cmd = ulg_in_df[UlgInDfKeys.p_rate_cmd].values
            q_rate_cmd = ulg_in_df[UlgInDfKeys.q_rate_cmd].values
            r_rate_cmd = ulg_in_df[UlgInDfKeys.r_rate_cmd].values
            az_cmd = ulg_in_df[UlgInDfKeys.az_cmd].values
            x_mixer_in = np.array([p_rate_cmd, q_rate_cmd, r_rate_cmd, az_cmd])
            x_mixer_in = np.transpose(x_mixer_in)

            x_nsxnv = x_mixer_in
            y_nsx1 = arm_df[arm_key_mi.pow_esc].values
            if x_nsxnv.shape[0] == 0:
                x_nsxnv = np.array([
                    [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
                y_nsx1 = np.array([0, 0, 0, 0])
                print('x_nsxnv = x_mixer_in')
                print('x_nsxnv.shape[0] == 0')
            linreg_fit = linreg.fit(x_nsxnv, y_nsx1)
            linreg_a = linreg_fit.coef_
            linreg_b = linreg_fit.intercept_

            linreg_dict[f'm{mi}_linreg'] = {
                'linreg_fit': linreg_fit,
                'linreg_a': linreg_a,
                'linreg_b': linreg_b,
            }

        # print(linreg_dict)
        return linreg_dict

    @staticmethod
    def linreg_powest_v2(arm_df, firefly_df, ulg_dict):
        firefly_df = copy.deepcopy(firefly_df)
        arm_df = copy.deepcopy(arm_df)
        ulg_dict = copy.deepcopy(ulg_dict)

        # https://scikit-learn.org/stable/modules/generated/
        # sklearn.linear_model.LinearRegression.html
        linreg = LinearRegression()
        # y_hat = lreg_a * x + lreg_b

        linreg_dict = {}
        for mi in ['16', '25', '38', '47']:
            arm_key_mi = ArmDfKeysMi(mi)
            mask = (arm_df[arm_key_mi.eta_rpm] < 1.1) & (
                    arm_df[arm_key_mi.eta_rpm] > 0.9)
            mask = pandas.Series(mask)
            mask.index = firefly_df.index
            [firefly_df, arm_df, ulg_dict] = FUParser.filter_by_mask(
                mask, firefly_df, arm_df, ulg_dict)

            ff_key_mu = FireflyDfKeysMi(mi[0])
            ff_key_ml = FireflyDfKeysMi(mi[1])
            upper_rpm = firefly_df[ff_key_mu.rpm].values
            lower_rpm = firefly_df[ff_key_ml.rpm].values
            x_rpm = np.array([upper_rpm, lower_rpm])
            x_rpm = np.transpose(x_rpm)

            x_nsxnv = x_rpm
            y_nsx1 = arm_df[arm_key_mi.pow_esc].values
            if x_nsxnv.shape[0] == 0:
                x_nsxnv = np.array([[0, 0], [0, 0]])
                y_nsx1 = np.array([0, 0])
                print('x_nsxnv = x_rpm')
                print('x_nsxnv.shape[0] == 0')
            linreg_fit = linreg.fit(x_nsxnv, y_nsx1)
            linreg_a = linreg_fit.coef_
            linreg_b = linreg_fit.intercept_

            linreg_dict[f'm{mi}_linreg'] = {
                'linreg_fit': linreg_fit,
                'linreg_a': linreg_a,
                'linreg_b': linreg_b,
            }

        # print(linreg_dict)
        return linreg_dict

    @staticmethod
    def calculate_nsh_cmd(firefly_df, arm_df, ulg_dict):
        _ = firefly_df, arm_df

        ulg_in_df = ulg_dict[UlgDictKeys.ulg_in_df]
        p_rate_cmd = ulg_in_df[UlgInDfKeys.p_rate_cmd].values
        q_rate_cmd = ulg_in_df[UlgInDfKeys.q_rate_cmd].values
        r_rate_cmd = ulg_in_df[UlgInDfKeys.r_rate_cmd].values
        az_cmd = ulg_in_df[UlgInDfKeys.az_cmd].values
        x_in = np.array([p_rate_cmd, q_rate_cmd, r_rate_cmd, az_cmd])
        # x_nvxns = x_in
        # x_in = np.transpose(x_in)
        # x_nsxnv = x_in

        ulg_out_df = ulg_dict[UlgDictKeys.ulg_out_df]
        out_0 = (ulg_out_df[UlgOutDfKeys.output_0] - 1500) / 500
        out_1 = (ulg_out_df[UlgOutDfKeys.output_1] - 1500) / 500
        out_2 = (ulg_out_df[UlgOutDfKeys.output_2] - 1500) / 500
        out_3 = (ulg_out_df[UlgOutDfKeys.output_3] - 1500) / 500
        out_4 = (ulg_out_df[UlgOutDfKeys.output_4] - 1500) / 500
        out_5 = (ulg_out_df[UlgOutDfKeys.output_5] - 1500) / 500
        out_6 = (ulg_out_df[UlgOutDfKeys.output_6] - 1500) / 500
        out_7 = (ulg_out_df[UlgOutDfKeys.output_7] - 1500) / 500

        x_out = np.array(
            [out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7])
        # x_nvxns = x_out
        # x_out = np.transpose(x_out)
        # x_nsxnv = x_out

        # a = 0  1
        #     2  3
        # b = 4  5
        #     6  7
        # c = a*b = 6  7
        #           26 31
        # a = np.array([[0, 1], [2, 3]])
        # b = np.array([[4, 5], [6, 7]])
        # c = np.matmul(a, b)
        # array([[ 6,  7], [26, 31]])

        # Octorotor Coaxial
        # Obtained from ctrlalloc_octocoax_px4.m
        b8_pinvn = [
            [-1.4142, +1.4142, +2.0000, +2.0000, +0.4981, +0.0019, -0.0019,
             +0.0019],
            [+1.4142, +1.4142, -2.0000, +2.0000, +0.0019, +0.4981, +0.0019,
             -0.0019],
            [+1.4142, -1.4142, +2.0000, +2.0000, -0.0019, +0.0019, +0.4981,
             +0.0019],
            [-1.4142, -1.4142, -2.0000, +2.0000, +0.0019, -0.0019, +0.0019,
             +0.4981],
            [+1.4142, +1.4142, +2.0000, +2.0000, -0.0019, -0.4981, -0.0019,
             +0.0019],
            [-1.4142, +1.4142, -2.0000, +2.0000, -0.4981, -0.0019, +0.0019,
             -0.0019],
            [-1.4142, -1.4142, +2.0000, +2.0000, -0.0019, +0.0019, -0.0019,
             -0.4981],
            [+1.4142, -1.4142, -2.0000, +2.0000, +0.0019, -0.0019, -0.4981,
             -0.0019],
        ]
        b8 = np.linalg.pinv(b8_pinvn)
        x_in_hat = []
        for i in range(0, x_out.shape[1]):  # number of columns
            thr = x_out[:, i]
            cmd = np.matmul(b8, thr)
            x_in_hat.append(cmd)
        x_in_hat = np.transpose(np.array(x_in_hat))
        # x_nvxns = x_in_hat
        x_in_hat[3, :] = x_in_hat[3, :] + 0.5
        nsh_cmd = np.array(copy.deepcopy(x_in_hat[4:8, :]))
        return [x_in, x_in_hat, nsh_cmd]

    @staticmethod
    def load_kdecf245dp_35_2p0_0p9_bet_df():
        # betfile = '/home/tzo4/Dropbox/tomas/pennState_avia/software/' \
        #           'static_MT_BET_analysis/python_code/img/' \
        #           'img_collective_roar/'
        betfile = 'bet_coax_eta_thrust_KDECF245DP_35_2.0_0.9.txt'

        bet_df = pandas.read_csv(betfile, delimiter=',', float_precision='high')
        rn_dict = {
            'Var1': 'bet_m1_thrust',
            'Var2': 'bet_m2_thrust',
            'Var3': 'bet_eta_thrust',
            'Var4': 'bet_m1_omega',
            'Var5': 'bet_m2_omega',
            'Var6': 'bet_eta_omega',
            'Var7': 'bet_m1_power',
            'Var8': 'bet_m2_power',
            'Var9': 'bet_coax_power',
        }
        bet_df = bet_df.rename(columns=rn_dict)

        rads_to_rpm = 30 / np.pi
        domega = bet_df['bet_m1_omega'].values - bet_df['bet_m2_omega'].values
        bet_df['bet_delta_rpm'] = domega * rads_to_rpm

        return bet_df

    @staticmethod
    def calculate_nshstats_df(firefly_df, arm_df, ulg_dict, file_tag):
        nsh_dict = FUParser.calculate_nsh_dict(
            firefly_df, arm_df, ulg_dict, file_tag, False)

        new_dict = {}
        cmd_arr = []
        for nsh_key in nsh_dict.keys():
            nsh_data = nsh_dict[nsh_key]

            split_arr = nsh_key.split('_')
            # nsh_key = f'i0_{i0}_i1_{i1}_nsh_{nsh_cmd}'
            # split_arr = 0   1   2   3   4   5
            # i0 = int(split_arr[1])
            # i1 = int(split_arr[3])
            cmd = float(split_arr[5])
            # print(f'i0 {i0}, i1 {i1}, cmd {cmd}')

            cmd_arr.append(cmd)

            for nsh_data_key in nsh_data.keys():
                df = nsh_data[nsh_data_key]
                df_mean = np.mean(df.values)
                df_std = np.std(df.values)

                key_mean = f'{nsh_data_key}_mean'
                key_std = f'{nsh_data_key}_std'

                try:
                    new_dict[key_mean].append(df_mean)
                    new_dict[key_std].append(df_std)
                except KeyError:
                    new_dict[key_mean] = [df_mean]
                    new_dict[key_std] = [df_std]

                if nsh_data_key == 'm16_pow_esc':
                    t0 = df.index[0]
                    t1 = df.index[-1]
                    key_time0 = 'time0'
                    key_time1 = 'time1'
                    try:
                        new_dict[key_time0].append(t0)
                        new_dict[key_time1].append(t1)
                    except KeyError:
                        new_dict[key_time0] = [t0]
                        new_dict[key_time1] = [t1]

                # # new_nsh_dict[nsh_data_key].append(val)
                # print(f'nsh_dict.keys() {nsh_dict.keys()}')
                # print(f'nsh_data.keys() {nsh_data.keys()}')
                # print(f'nsh_key {nsh_key}')
                # print(f'nsh_data_key {nsh_data_key}')
                # print(f'nsh_data[nsh_data_key] {nsh_data[nsh_data_key]}')
                # exit(0)

        new_dict['delta_cmd'] = cmd_arr
        cmd_drpm = 1650 * np.array(cmd_arr)
        new_dict['cmd_delta_rpm'] = cmd_drpm
        new_dict['nsh_dict_keys'] = nsh_dict.keys()
        m16_devsq = (np.array(new_dict['m16_delta_rpm_mean']) - cmd_drpm)**2
        m25_devsq = (np.array(new_dict['m25_delta_rpm_mean']) - cmd_drpm)**2
        m38_devsq = (np.array(new_dict['m38_delta_rpm_mean']) - cmd_drpm)**2
        m47_devsq = (np.array(new_dict['m47_delta_rpm_mean']) - cmd_drpm)**2
        sum_devsq = m16_devsq + m25_devsq + m38_devsq + m47_devsq
        new_dict['rmsd_delta_rpm'] = np.sqrt(sum_devsq / 4)
        # print(f'new_nsh_dict.keys() {new_nsh_dict.keys()}')

        nshstats_df = pandas.DataFrame.from_dict(
            data=new_dict, orient='columns', dtype=None, columns=None)
        # print(nshstats_df)
        # Remove repeated delta_cmd with highest m16_pow_esc_std
        # iarr = [0, 1, 2, 3, 4, 11, 12, 13, 14]
        # nshstats_df.drop(index=iarr, inplace=True)
        # nshstats_df.index = nshstats_df['delta_cmd']
        # nshstats_df.drop(columns='delta_cmd', inplace=True)

        return nshstats_df

    @staticmethod
    def calculate_nsh_dict(firefly_df, arm_df, ulg_dict, file_tag, verbose):
        # data_dict = FileTagData.data_dict(file_tag)
        if (file_tag == 'firefly_log_54_ulg_log_109') or \
                (file_tag == 'firefly_log_32_ulg_log_105'):
            nsh_dict = FUParser.nsh_dict_for_ulg_log_105_and_109(
                firefly_df, arm_df, ulg_dict)
        elif file_tag == 'firefly_log_7_ulg_log_137':
            nsh_dict = FUParser.nsh_dict_for_ulg_log_145(
                firefly_df, arm_df, ulg_dict)
        elif file_tag == 'firefly_log_5_ulg_log_143':
            nsh_dict = FUParser.nsh_dict_for_ulg_log_145(
                firefly_df, arm_df, ulg_dict)
        elif file_tag == 'firefly_log_8_ulg_log_144':
            nsh_dict = FUParser.nsh_dict_for_ulg_log_145(
                firefly_df, arm_df, ulg_dict)
        elif file_tag == 'firefly_log_9_ulg_log_145':
            nsh_dict = FUParser.nsh_dict_for_ulg_log_145(
                firefly_df, arm_df, ulg_dict)
        else:
            raise RuntimeError
        if verbose:
            print(f'[calculate_nsh_dict] nsh_dict.keys() {nsh_dict.keys()}')
        return nsh_dict

    @staticmethod
    def nsh_dict_for_ulg_log_145(firefly_df, arm_df, ulg_dict):
        # print('[nsh_dict_for_ulg_log_109] Calculating nsh windows ..')

        ulg_in_df = ulg_dict[UlgDictKeys.ulg_in_df]
        ulg_pv_df = ulg_dict[UlgDictKeys.ulg_pv_df]
        ulg_angvel_df = ulg_dict[UlgDictKeys.ulg_angvel_df]

        # [x_in, x_in_hat, nsh_cmd] = FUParser.calculate_nsh_cmd(
        #     firefly_df, arm_df, ulg_dict)
        # _ = x_in, x_in_hat
        nsh_cmd = firefly_df[FireflyDfKeys.nsh_cmd]
        firefly_df[FireflyDfKeys.nsh_cmd] = np.around(
            nsh_cmd.values * 2, decimals=1) / 2
        nsh_cmd = firefly_df[FireflyDfKeys.nsh_cmd]
        # print(f'nsh_cmd {nsh_cmd}')

        # for i in range(0, int(len(nsh_cmd.index) / 10)):
        #     print(f'nsh_cmd.index[{i}] {nsh_cmd.index[i]} '
        #           f'nsh_cmd.iloc[{i}] {nsh_cmd.iloc[i]}')
        # exit(0)
        nsh_cmd = nsh_cmd.values

        m16_pow_esc = arm_df[ArmDfKeys.m16.pow_esc]
        m25_pow_esc = arm_df[ArmDfKeys.m25.pow_esc]
        m38_pow_esc = arm_df[ArmDfKeys.m38.pow_esc]
        m47_pow_esc = arm_df[ArmDfKeys.m47.pow_esc]
        tot_pow_esc = m16_pow_esc + m25_pow_esc + m38_pow_esc + m47_pow_esc

        m38_pow_ars = arm_df[ArmDfKeys.m38.pow_ars]
        m47_pow_ars = arm_df[ArmDfKeys.m47.pow_ars]

        m16_delta_rpm = arm_df[ArmDfKeys.m16.delta_rpm]
        m25_delta_rpm = arm_df[ArmDfKeys.m25.delta_rpm]
        m38_delta_rpm = arm_df[ArmDfKeys.m38.delta_rpm]
        m47_delta_rpm = arm_df[ArmDfKeys.m47.delta_rpm]
        net_delta_rpm = (m16_delta_rpm - m25_delta_rpm +
                         m38_delta_rpm - m47_delta_rpm)
        mean_delta_rpm = (m16_delta_rpm + m25_delta_rpm +
                          m38_delta_rpm + m47_delta_rpm) / 4

        m1_rpm = firefly_df[FireflyDfKeys.m1.rpm]
        m2_rpm = firefly_df[FireflyDfKeys.m2.rpm]
        m3_rpm = firefly_df[FireflyDfKeys.m3.rpm]
        m4_rpm = firefly_df[FireflyDfKeys.m4.rpm]
        m5_rpm = firefly_df[FireflyDfKeys.m5.rpm]
        m6_rpm = firefly_df[FireflyDfKeys.m6.rpm]
        m7_rpm = firefly_df[FireflyDfKeys.m7.rpm]
        m8_rpm = firefly_df[FireflyDfKeys.m8.rpm]
        mean_tot_rpm = (m1_rpm + m2_rpm + m3_rpm + m4_rpm +
                        m5_rpm + m6_rpm + m7_rpm + m8_rpm) / 8

        m16_eta_rpm = arm_df[ArmDfKeys.m16.eta_rpm]
        m25_eta_rpm = arm_df[ArmDfKeys.m25.eta_rpm]
        m38_eta_rpm = arm_df[ArmDfKeys.m38.eta_rpm]
        m47_eta_rpm = arm_df[ArmDfKeys.m47.eta_rpm]
        mean_eta_rpm = (m16_eta_rpm + m25_eta_rpm +
                        m38_eta_rpm + m47_eta_rpm) / 4

        nsh_decimals = 2
        nsh_min_diff = 0.03

        # nsh_arr = np.around(nsh_arr, decimals=nsh_decimals)
        # nsh_arr = np.around(nsh_arr / 0.05, decimals=0) * 0.05
        # round(number * 2) / 2

        nsh_diff = np.diff(nsh_cmd)
        nsh_diff = np.insert(nsh_diff, obj=0, values=nsh_diff[0])
        indx_nsh_change = np.argwhere(np.abs(nsh_diff) > nsh_min_diff)
        indx_nsh_change = indx_nsh_change.squeeze()
        # indx_nsh_change = indx_nsh_change.append(len(nsh_arr) - 1)
        if len(indx_nsh_change) == 0:
            # there were no changes in nsh_cmd
            indx_nsh_change = [len(nsh_diff) - 1]
        # print(f'indx_nsh_change {indx_nsh_change}')
        # print(f'nsh_arr[indx_nsh_change] {nsh_cmd[indx_nsh_change]}')

        # nsh_dict = {'nsh_arr': nsh_arr, 'nsh_diff': nsh_diff}
        nsh_dict = {}
        prev_indx = 0
        for indx in indx_nsh_change:
            i0 = prev_indx
            i1 = indx - 1
            t0 = round(m16_pow_esc.index[i0], 2)
            t1 = round(m16_pow_esc.index[i1], 2)
            if (t1-t0) < 1:
                print(f'[nsh_dict_for_ulg_log_145] (t1-t0) < 3')
                print(f'[nsh_dict_for_ulg_log_145] ({t1}-{t0}) < 3')

            # windows is from i0 to i1 (inclusive), thus range is [i0:i1+1]
            i2 = i1 + 1

            nsh_data = {
                'm16_pow_esc': m16_pow_esc.iloc[i0:i2],
                'm25_pow_esc': m25_pow_esc.iloc[i0:i2],
                'm38_pow_esc': m38_pow_esc.iloc[i0:i2],
                'm47_pow_esc': m47_pow_esc.iloc[i0:i2],
                'tot_pow_esc': tot_pow_esc.iloc[i0:i2],
                'm38_pow_ars': m38_pow_ars.iloc[i0:i2],
                'm47_pow_ars': m47_pow_ars.iloc[i0:i2],

                'm16_delta_rpm': m16_delta_rpm.iloc[i0:i2],
                'm25_delta_rpm': m25_delta_rpm.iloc[i0:i2],
                'm38_delta_rpm': m38_delta_rpm.iloc[i0:i2],
                'm47_delta_rpm': m47_delta_rpm.iloc[i0:i2],
                'net_delta_rpm': net_delta_rpm.iloc[i0:i2],
                'mean_delta_rpm': mean_delta_rpm.iloc[i0:i2],

                'mean_tot_rpm': mean_tot_rpm.iloc[i0:i2],

                'm16_eta_rpm': m16_eta_rpm.iloc[i0:i2],
                'm25_eta_rpm': m25_eta_rpm.iloc[i0:i2],
                'm38_eta_rpm': m38_eta_rpm.iloc[i0:i2],
                'm47_eta_rpm': m47_eta_rpm.iloc[i0:i2],
                'mean_eta_rpm': mean_eta_rpm.iloc[i0:i2],

                'r_rate_cmd': ulg_in_df[UlgInDfKeys.r_rate_cmd].iloc[i0:i2],
                'vel_norm': ulg_pv_df[UlgPvDfKeys.vel_norm].iloc[i0:i2],
                'pqr_norm': ulg_angvel_df[UlgAngvelDf.pqr_norm].iloc[i0:i2],
            }

            nsh_window = nsh_cmd[i0:i2]
            cmd = round(np.mean(nsh_window), nsh_decimals)
            # nsh_key = f'i0_{i0}_i1_{i1}_t0_{t0}_t1_{t1}_nsh_{nsh}'
            nsh_key = f'i0_{i0}_i1_{i1}_cmd_{cmd}'
            nsh_dict[nsh_key] = copy.deepcopy(nsh_data)
            # split_arr = nsh_key.split('_')
            # print(f' '.join(split_arr))
            # print(nsh_key)

            prev_indx = indx

        # print(m16_pow_esc.shape)
        # print(nsh_dict['nshpow 0.0'])

        return nsh_dict

    @staticmethod
    def nsh_dict_for_ulg_log_105_and_109(firefly_df, arm_df, ulg_dict):
        # print('[nsh_dict_for_ulg_log_109] Calculating nsh windows ..')

        ulg_in_df = ulg_dict[UlgDictKeys.ulg_in_df]
        ulg_pv_df = ulg_dict[UlgDictKeys.ulg_pv_df]
        ulg_angvel_df = ulg_dict[UlgDictKeys.ulg_angvel_df]

        [x_in, x_in_hat, nsh_cmd] = FUParser.calculate_nsh_cmd(
            firefly_df, arm_df, ulg_dict)
        _ = x_in, x_in_hat

        m16_pow_esc = arm_df[ArmDfKeys.m16.pow_esc]
        m25_pow_esc = arm_df[ArmDfKeys.m25.pow_esc]
        m38_pow_esc = arm_df[ArmDfKeys.m38.pow_esc]
        m47_pow_esc = arm_df[ArmDfKeys.m47.pow_esc]
        tot_pow_esc = m16_pow_esc + m25_pow_esc + m38_pow_esc + m47_pow_esc
        # print(f"tot_pow_esc {tot_pow_esc}")

        m38_pow_ars = arm_df[ArmDfKeys.m38.pow_ars]
        m47_pow_ars = arm_df[ArmDfKeys.m47.pow_ars]

        m16_delta_rpm = arm_df[ArmDfKeys.m16.delta_rpm]
        m25_delta_rpm = arm_df[ArmDfKeys.m25.delta_rpm]
        m38_delta_rpm = arm_df[ArmDfKeys.m38.delta_rpm]
        m47_delta_rpm = arm_df[ArmDfKeys.m47.delta_rpm]
        net_delta_rpm = (m16_delta_rpm - m25_delta_rpm +
                         m38_delta_rpm - m47_delta_rpm)
        mean_delta_rpm = (m16_delta_rpm + m25_delta_rpm +
                          m38_delta_rpm + m47_delta_rpm) / 4

        m1_rpm = firefly_df[FireflyDfKeys.m1.rpm]
        m2_rpm = firefly_df[FireflyDfKeys.m2.rpm]
        m3_rpm = firefly_df[FireflyDfKeys.m3.rpm]
        m4_rpm = firefly_df[FireflyDfKeys.m4.rpm]
        m5_rpm = firefly_df[FireflyDfKeys.m5.rpm]
        m6_rpm = firefly_df[FireflyDfKeys.m6.rpm]
        m7_rpm = firefly_df[FireflyDfKeys.m7.rpm]
        m8_rpm = firefly_df[FireflyDfKeys.m8.rpm]
        mean_tot_rpm = (m1_rpm + m2_rpm + m3_rpm + m4_rpm +
                        m5_rpm + m6_rpm + m7_rpm + m8_rpm) / 8

        m16_eta_rpm = arm_df[ArmDfKeys.m16.eta_rpm]
        m25_eta_rpm = arm_df[ArmDfKeys.m25.eta_rpm]
        m38_eta_rpm = arm_df[ArmDfKeys.m38.eta_rpm]
        m47_eta_rpm = arm_df[ArmDfKeys.m47.eta_rpm]

        nsh_decimals = 1
        nsh_min_diff = 0.03

        nsh_i = 0
        nsh_arr = nsh_cmd[nsh_i, :]
        nsh_arr = np.around(nsh_arr, decimals=nsh_decimals)
        # nsh_arr = np.around(nsh_arr / 0.05, decimals=0) * 0.05
        # round(number * 2) / 2

        nsh_diff = np.diff(nsh_arr)
        nsh_diff = np.insert(nsh_diff, obj=0, values=nsh_diff[0])
        indx_nsh_change = np.argwhere(np.abs(nsh_diff) > nsh_min_diff)
        indx_nsh_change = indx_nsh_change.squeeze()
        # indx_nsh_change = indx_nsh_change.append(len(nsh_arr) - 1)
        if len(indx_nsh_change) == 0:
            # there were no changes in nsh_cmd
            indx_nsh_change = [len(nsh_diff)-1]
        # print(f'indx_nsh_change {indx_nsh_change}')
        # print(f'nsh_arr[indx_nsh_change] {nsh_arr[indx_nsh_change]}')

        # nsh_dict = {'nsh_arr': nsh_arr, 'nsh_diff': nsh_diff}
        nsh_dict = {}
        prev_indx = 0
        for indx in indx_nsh_change:
            i0 = prev_indx
            i1 = indx - 1
            t0 = round(m16_pow_esc.index[i0], 2)
            t1 = round(m16_pow_esc.index[i1], 2)
            if (t1-t0) < 3:
                print(f'[nsh_dict_for_ulg_log_105_and_109] (t1-t0) < 3')
                print(f'[nsh_dict_for_ulg_log_105_and_109] ({t1}-{t0}) < 3')

            # windows is from i0 to i1 (inclusive), thus range is [i0:i1+1]
            i2 = i1 + 1

            nsh_data = {
                'm16_pow_esc': m16_pow_esc.iloc[i0:i2],
                'm25_pow_esc': m25_pow_esc.iloc[i0:i2],
                'm38_pow_esc': m38_pow_esc.iloc[i0:i2],
                'm47_pow_esc': m47_pow_esc.iloc[i0:i2],
                'tot_pow_esc': tot_pow_esc.iloc[i0:i2],
                'm38_pow_ars': m38_pow_ars.iloc[i0:i2],
                'm47_pow_ars': m47_pow_ars.iloc[i0:i2],

                'm16_delta_rpm': m16_delta_rpm.iloc[i0:i2],
                'm25_delta_rpm': m25_delta_rpm.iloc[i0:i2],
                'm38_delta_rpm': m38_delta_rpm.iloc[i0:i2],
                'm47_delta_rpm': m47_delta_rpm.iloc[i0:i2],
                'net_delta_rpm': net_delta_rpm.iloc[i0:i2],
                'mean_delta_rpm': mean_delta_rpm.iloc[i0:i2],

                'mean_tot_rpm': mean_tot_rpm.iloc[i0:i2],

                'm16_eta_rpm': m16_eta_rpm.iloc[i0:i2],
                'm25_eta_rpm': m25_eta_rpm.iloc[i0:i2],
                'm38_eta_rpm': m38_eta_rpm.iloc[i0:i2],
                'm47_eta_rpm': m47_eta_rpm.iloc[i0:i2],

                'r_rate_cmd': ulg_in_df[UlgInDfKeys.r_rate_cmd].iloc[i0:i2],
                'vel_norm': ulg_pv_df[UlgPvDfKeys.vel_norm].iloc[i0:i2],
                'pqr_norm': ulg_angvel_df[UlgAngvelDf.pqr_norm].iloc[i0:i2],
            }

            nsh_window = nsh_arr[i0:i2]
            cmd = round(np.mean(nsh_window), nsh_decimals)
            # nsh_key = f'i0_{i0}_i1_{i1}_t0_{t0}_t1_{t1}_nsh_{nsh}'
            nsh_key = f'i0_{i0}_i1_{i1}_cmd_{cmd}'
            nsh_dict[nsh_key] = copy.deepcopy(nsh_data)
            # split_arr = nsh_key.split('_')
            # print(f' '.join(split_arr))
            # print(nsh_key)

            prev_indx = indx

        # print(m16_pow_esc.shape)
        # print(nsh_dict['nshpow 0.0'])

        return nsh_dict

    @staticmethod
    def calculate_powstat_dict(arm_df):
        # calculate_power_by_x_ranges
        m16_eta_rpm = arm_df[ArmDfKeys.m16.eta_rpm].values
        m25_eta_rpm = arm_df[ArmDfKeys.m25.eta_rpm].values
        m38_eta_rpm = arm_df[ArmDfKeys.m38.eta_rpm].values
        m47_eta_rpm = arm_df[ArmDfKeys.m47.eta_rpm].values

        m16_delta_rpm = arm_df[ArmDfKeys.m16.delta_rpm].values
        m25_delta_rpm = arm_df[ArmDfKeys.m25.delta_rpm].values
        m38_delta_rpm = arm_df[ArmDfKeys.m38.delta_rpm].values
        m47_delta_rpm = arm_df[ArmDfKeys.m47.delta_rpm].values

        m16_pow_esc = arm_df[ArmDfKeys.m16.pow_esc].values
        m25_pow_esc = arm_df[ArmDfKeys.m25.pow_esc].values
        m38_pow_esc = arm_df[ArmDfKeys.m38.pow_esc].values
        m47_pow_esc = arm_df[ArmDfKeys.m47.pow_esc].values

        min_ns = 10
        stat_dict = {}

        dx = 100
        delta_rpm = np.arange(-1400 + dx / 2, +1500 + dx, dx)
        # print(f'delta_rpm {delta_rpm}')

        xkey_arr = [
            'm16_delta_rpm', 'm25_delta_rpm', 'm38_delta_rpm', 'm47_delta_rpm']
        xvar_arr = [m16_delta_rpm, m25_delta_rpm, m38_delta_rpm, m47_delta_rpm]
        ykey_arr = [
            'm16_pow_esc', 'm25_pow_esc', 'm38_pow_esc', 'm47_pow_esc']
        yvar_arr = [m16_pow_esc, m25_pow_esc, m38_pow_esc, m47_pow_esc]
        for x_arr, y_arr, x_key, y_key in zip(
                xvar_arr, yvar_arr, xkey_arr, ykey_arr):
            x_mean, x_std, y_mean, y_std = FUParser.stats_by_index_of_x_arr(
                y_arr=y_arr, x_arr=x_arr, x_ranges=delta_rpm, min_samples=min_ns)
            stat_dict[f'{x_key}_mean'] = x_mean
            stat_dict[f'{x_key}_std'] = x_std
            stat_dict[f'{x_key}_mean_{y_key}_mean'] = y_mean
            stat_dict[f'{x_key}_mean_{y_key}_std'] = y_std

        dx = 0.1 / 2
        eta_rpm = np.arange(dx + dx / 2, 2.0 + dx, dx)
        # eta_rpm = eta_rpm - dx
        # print(f'eta_rpm {eta_rpm}')

        xkey_arr = [
            'm16_eta_rpm', 'm25_eta_rpm', 'm38_eta_rpm', 'm47_eta_rpm']
        xvar_arr = [m16_eta_rpm, m25_eta_rpm, m38_eta_rpm, m47_eta_rpm]
        ykey_arr = [
            'm16_pow_esc', 'm25_pow_esc', 'm38_pow_esc', 'm47_pow_esc']
        yvar_arr = [m16_pow_esc, m25_pow_esc, m38_pow_esc, m47_pow_esc]
        for x_arr, y_arr, x_key, y_key in zip(
                xvar_arr, yvar_arr, xkey_arr, ykey_arr):
            x_mean, x_std, y_mean, y_std = FUParser.stats_by_index_of_x_arr(
                y_arr=y_arr, x_arr=x_arr, x_ranges=eta_rpm, min_samples=min_ns)

            stat_dict[f'{x_key}_mean'] = x_mean
            stat_dict[f'{x_key}_std'] = x_std
            stat_dict[f'{x_key}_mean_{y_key}_mean'] = y_mean
            stat_dict[f'{x_key}_mean_{y_key}_std'] = y_std

        # print(stat_dict.keys())
        return stat_dict

    @staticmethod
    def stats_by_index_of_x_arr(y_arr, x_arr, x_ranges, min_samples):
        val_dict = {}
        x_prev = x_ranges[0]
        for x in x_ranges[1:]:
            mask = (x_arr > x_prev) & (x_arr < x)
            indx = np.argwhere(mask)
            # print(f'x_prev {x_prev} x {x}')
            # print(f'mask {mask}, indx {indx}')
            if len(indx) > min_samples:
                # min_val = np.min(val_arr[indx])
                # max_val = np.max(val_arr[indx])
                # print(f'min_val {min_val}, max_val {max_val}')
                x_mean = np.mean(x_arr[indx])
                x_std = np.std(x_arr[indx])
                y_mean = np.mean(y_arr[indx])
                y_std = np.std(y_arr[indx])
            else:
                # x_mean = np.nan
                # x_std = np.nan
                # y_mean = np.nan
                # y_std = np.nan
                continue

            try:
                val_dict[f'x_mean'].append(x_mean)
                val_dict[f'x_std'].append(x_std)
                val_dict[f'y_mean'].append(y_mean)
                val_dict[f'y_std'].append(y_std)
            except KeyError:
                val_dict[f'x_mean'] = [x_mean]
                val_dict[f'x_std'] = [x_std]
                val_dict[f'y_mean'] = [y_mean]
                val_dict[f'y_std'] = [y_std]
            x_prev = x
        x_mean = val_dict[f'x_mean']
        x_std = val_dict[f'x_std']
        y_mean = val_dict[f'y_mean']
        y_std = val_dict[f'y_std']
        return x_mean, x_std, y_mean, y_std

    @staticmethod
    def print_latex_table_total_power(nshstats_df, ref_tpe_mean):
        # rn_dict = {
        #     'delta_cmd': 'delta_cmd',
        #     'cmd_delta_rpm': 'cmd_drpm',
        #     # 'm16_delta_rpm_mean': 'm16_drpm_mean',
        #     # 'm25_delta_rpm_mean': 'm25_drpm_mean',
        #     # 'm38_delta_rpm_mean': 'm38_drpm_mean',
        #     # 'm47_delta_rpm_mean': 'm47_drpm_mean',
        #     'mean_delta_rpm_mean': 'mean_drpm_mean',
        #     'rmsd_delta_rpm': 'rmsd_drpm',
        #     'net_delta_rpm_mean': 'net_drpm_mean',
        #     'tot_pow_esc_mean': 'tpow_mean',
        #     'tot_pow_esc_std': 'tpow_std',
        #     'm38_pow_ars_mean': 'm38_pow_ars',
        # }
        for ind in nshstats_df.index:
            delta_cmd = round(nshstats_df['delta_cmd'][ind], 2)
            cmd_drpm = round(nshstats_df['cmd_delta_rpm'][ind], 2)
            mdr_mean = round(nshstats_df['mean_delta_rpm_mean'][ind])
            rdr_mean = round(nshstats_df['rmsd_delta_rpm'][ind])
            tpe_mean = round(nshstats_df['tot_pow_esc_mean'][ind])
            tpe_perc = round((tpe_mean - ref_tpe_mean) / ref_tpe_mean * 100, 1)
            tpe_std = round(nshstats_df['tot_pow_esc_std'][ind])

            # net_rpm = round(nshstats_df['net_delta_rpm_mean'][ind])
            # mtr_mean = round(nshstats_df['mean_tot_rpm_mean'][ind])
            # mtr_perc = round((mtr_mean - ref_mtr_mean)/ref_mtr_mean*100, 1)
            # latex_perc = r'\%'
            # print(f'{delta_cmd} & {tpe_mean} ({tpe_perc} {latex_perc}) & '
            #       f'{tpe_std} & {mtr_mean} ({mtr_perc} {latex_perc}) ')

            # ndk = nshstats_df['nsh_dict_keys'][ind]
            #  {ndk}
            latex_perc = r'\%'
            print(f'{delta_cmd} & {cmd_drpm} & {mdr_mean} & {rdr_mean}'
                  f' & {tpe_mean} ({tpe_perc} {latex_perc}) & {tpe_std}')

        # firefly_log_54_ulg_log_109
        # 0.0 & 1308 (-0.1 \%) & 58 & 17658 (-0.1 \%)
        # 0.1 & 1336 (2.1 \%) & 21 & 17822 (0.8 \%)
        # 0.2 & 1364 (4.2 \%) & 12 & 17959 (1.6 \%)
        # 0.3 & 1386 (5.9 \%) & 22 & 18016 (1.9 \%)
        # 0.4 & 1425 (8.9 \%) & 19 & 18120 (2.5 \%)
        # 0.5 & 1448 (10.6 \%) & 16 & 18106 (2.4 \%)
        # 0.4 & 1419 (8.4 \%) & 17 & 18090 (2.3 \%)
        # 0.3 & 1386 (5.9 \%) & 15 & 18015 (1.9 \%)
        # 0.2 & 1365 (4.3 \%) & 27 & 17964 (1.6 \%)
        # 0.1 & 1346 (2.8 \%) & 21 & 17874 (1.1 \%)
        # 0.0 & 1309 (0.0 \%) & 12 & 17680 (0.0 \%)
        # -0.1 & 1304 (-0.4 \%) & 14 & 17560 (-0.7 \%)
        # -0.2 & 1298 (-0.8 \%) & 22 & 17394 (-1.6 \%)
        # -0.3 & 1286 (-1.8 \%) & 25 & 17163 (-2.9 \%)
        # -0.4 & 1296 (-1.0 \%) & 32 & 17013 (-3.8 \%)
        # -0.5 & 1295 (-1.1 \%) & 26 & 16759 (-5.2 \%)
        # -0.4 & 1292 (-1.3 \%) & 24 & 17007 (-3.8 \%)
        # -0.3 & 1279 (-2.3 \%) & 16 & 17152 (-3.0 \%)
        # -0.2 & 1300 (-0.7 \%) & 18 & 17422 (-1.5 \%)
        # -0.1 & 1305 (-0.3 \%) & 29 & 17563 (-0.7 \%)


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
    parser.add_argument('--ulg', action='store', required=True,
                        help='Specific log file number to process')
    parser.add_argument('--firefly', action='store', required=True,
                        help='Specific log file number to process')

    args = parser.parse_args()

    abs_bdir = os.path.abspath(args.bdir)
    if args.firefly is not None:
        abs_firefly_file = find_file_in_folder(
            f'{abs_bdir}/logs', '.firefly', args.firefly)
        abs_ulg_file = find_file_in_folder(
            f'{abs_bdir}/logs', '.ulg', args.ulg)
        print(f'ulg file: {abs_ulg_file}')
        print(f'firefly file: {abs_firefly_file}')

        f_parser = FUParser(abs_bdir, abs_firefly_file, abs_ulg_file)
        exit(0)

    print('Error parsing user arguments')
    print(f'firefly file: {args.firefly}')
    print(f'kdecan file: {args.kdecan}')
    print(f'ulg file: {args.ulg}')
    exit(0)

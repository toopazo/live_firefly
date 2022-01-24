import copy
import argparse
import pandas
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt

from toopazo_tools.pandas import DataframeTools
from toopazo_tools.file_folder import FileFolderTools as FFTools
from live_esc.kde_uas85uvc.kdecan_parse import KdecanParser
# from live_esc.kde_uas85uvc.kdecan_parse import EscidParserTools
from live_ars.ars_parse import ArsParser


class AKParser:
    def __init__(self, bdir, kdecan_file, ars_file):
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
            raise RuntimeError(f'[AKParser] No such file {kdecan_file}')
        else:
            self.kdecan_file = kdecan_file
            self.escid_dict = KdecanParser.get_escid_dict(kdecan_file)

        if not os.path.isfile(ars_file):
            raise RuntimeError(f'[AKParser] No such file {ars_file}')
        else:
            self.ars_file = ars_file
            self.ars_df = ArsParser.get_pandas_dataframe(ars_file)

        self.figsize = (10, 6)

        # print(self.escid_dict.keys())
        # for key, val in self.escid_dict.items():
        #     print(key)
        #     print(val)
        # print(self.ars_df)

    def write_firefly_log(self, firefly_log, firefly_name, file_tag):
        ars_df = self.ars_df
        escid_dict = self.escid_dict
        print(ars_df.keys())
        print(escid_dict.keys())

        desired_dt = 0.1
        escid_dict = AKParser.process_escid_dict(escid_dict, desired_dt)

        ars_df = AKParser.process_ars_df(ars_df, desired_dt)
        ars_df = AKParser.apply_delay_and_reset_index(ars_df, file_tag)

        kdecan_df = AKParser.escid_dict_to_kdecan_df(escid_dict)

        [kdecan_df, ars_df] = AKParser.apply_hover_times(
            kdecan_df, ars_df, file_tag)

        self.plot_current(
            kdecan_df, ars_df, file_tag, [f'to_firefly_{firefly_log}'])

        line_arr = AKParser.get_line_arr(kdecan_df, ars_df, file_tag)
        print(f'len(line_arr) {len(line_arr)}')
        print(f'Writing to file {firefly_name}')
        fd = open(firefly_name, 'w')
        fd.writelines(line_arr)
        fd.close()

    @staticmethod
    def get_line_arr(kdecan_df, ars_df, file_tag):
        # 200, 999000, 999.0, 999001,
        # 40, 0, 38, 40, 0, 0, 0, 0, 657, 625, 655, 631, 7, 9, 11, 12,
        # 2021-12-20 23:50:33.546752, 11, 28.88, 14.03, 2288.57, 13,
        # No warning, 1485.0, 55.0 ,
        # 2021-12-20 23:50:33.560435, 12, 29.08, 9.06, 2172.86, 13,
        # ...
        # 2021-12-20 23:50:33.616664, 18, 28.57, 4.52, 2172.86, 13,
        # No warning, 1430.0, 50.0 ,
        # [-1, -1, 18895.32, 18818.15, -1, -1, 17945.64, 17856.25],
        # 0, 0, 0, 0, 0

        print(ars_df)
        print(ars_df.keys())

        print(kdecan_df)
        print(kdecan_df.keys())

        ars_df.index = kdecan_df.index

        line_cnt = 0
        line_arr = []
        for ind in kdecan_df.index:
            sps_mills_secs_dtmills = [200, int(ind*1000), ind, 1000]
            rpm1234 = ars_df.loc[ind][['rpm_13', 'rpm_18', 'rpm_14', 'rpm_17']]
            l1 = ", ".join([str(e) for e in sps_mills_secs_dtmills])
            l2 = ", ".join([str(e) for e in rpm1234])
            l3 = ", ".join(['0', '0', '0', '0'])
            cur1234 = ars_df.loc[ind][['cur_13', 'cur_18', 'cur_14', 'cur_17']]
            l4 = ", ".join([str(e) for e in cur1234])
            l5 = ", ".join(['0', '0', '0', '0'])

            lmi_arr = []
            for mi in range(11, 19):
                lmi_arr.append(ind)
                lmi_arr.append(mi)
                lmi_arr.append(kdecan_df.loc[ind][f'esc{mi}_voltage'])
                lmi_arr.append(kdecan_df.loc[ind][f'esc{mi}_current'])
                lmi_arr.append(kdecan_df.loc[ind][f'esc{mi}_rpm'])
                lmi_arr.append(25)
                lmi_arr.append('No warning')
                lmi_arr.append(kdecan_df.loc[ind][f'esc{mi}_inthrottle'])
                lmi_arr.append(100)
            l6 = ", ".join([str(e) for e in lmi_arr])
            l7 = '[-1, -1, -1, -1, -1, -1, -1, -1]'
            l8 = '0, 0, 0, 0, 0'

            line = ", ".join([l1, l2, l3, l4, l5, l6, l7, l8])
            line = f"{line}\n"
            line_arr.append(copy.deepcopy(line))
            line_cnt = line_cnt + 1
            # if line_cnt <= 3:
            #     print(line)
            # else:
            #     break
        return line_arr

    @staticmethod
    def apply_hover_times(kdecan_df, ars_df, file_tag):
        [date0, date1] = ManualAdj.hover_time_window(file_tag)
        mask = (ars_df.index > date0) & (ars_df.index <= date1)
        mask = pandas.Series(mask)
        mask.index = ars_df.index
        # Now let us apply the mask
        mask.index = ars_df.index
        ars_df = ars_df[mask]

        [date0, date1] = ManualAdj.hover_time_window(file_tag)
        mask = (kdecan_df.index > date0) & (kdecan_df.index <= date1)
        mask = pandas.Series(mask)
        mask.index = kdecan_df.index
        # Now let us apply the mask
        mask.index = kdecan_df.index
        kdecan_df = kdecan_df[mask]

        return [kdecan_df, ars_df]

    @staticmethod
    def escid_dict_to_kdecan_df(escid_dict):
        kdecan_df = pandas.DataFrame(index=escid_dict['esc11_df'].index)

        for i in range(1, 9):
            escid_key = f'esc1{i}_df'
            escid_df = escid_dict[escid_key]
            mi = FireflyKeys(i)
            kdecan_df[mi.thr] = escid_df['inthtl us']
            kdecan_df[mi.vol] = escid_df['voltage V']
            kdecan_df[mi.cur] = escid_df['current A']
            kdecan_df[mi.rpm] = escid_df['angVel rpm']

        print('kdecan_df')
        print(kdecan_df)
        return kdecan_df

    def plot_current(self, kdecan_df, ars_df, file_tag, tag_arr):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)
        fig.suptitle('ars and kdecan current')

        m3 = FireflyKeys(3)

        ax1.grid(True)
        ax1.set_ylabel("m3 esc", color='red')
        ax1.plot(kdecan_df[m3.cur], color='red')

        ax1t = ax1.twinx()
        ax1t.set_ylabel("m3 ars", color='blue')
        ax1t.plot(ars_df['cur_13'], color='blue')
        ax1t.axes.xaxis.set_ticklabels([])

        kdecan_current = [
            'esc13_current', 'esc14_current', 'esc17_current', 'esc18_current'
        ]
        ars_throttle = [
            'cur_13', 'cur_14', 'cur_17', 'cur_18'
        ]

        ax2.grid(True)
        ax2.set_ylabel("esc", color='red')
        ax2.plot(kdecan_df[kdecan_current], color='red')

        ax2t = ax2.twinx()
        ax2t.set_ylabel("ars", color='blue')
        ax2t.plot(ars_df[ars_throttle], color='blue')
        # ars_df[ars_throttle].plot(ax=ax2t, grid=True).legend(
        #     loc='center left')
        ax2t.set_xlabel("Time, s")

        self.save_current_plot(file_tag, tag_arr=tag_arr, sep="_", ext='.png')

    def save_current_plot(self, file_tag, tag_arr, sep, ext):
        file_name = file_tag
        for tag in tag_arr:
            file_name = file_name + sep + str(tag)
        file_path = self.plotdir + f'/' + file_name + ext

        # plt.show()
        print(f'Saving file {file_path} ..')
        plt.savefig(file_path)
        # return file_path

    @staticmethod
    def process_ars_df(ars_df, desired_dt):
        ars_df.drop_duplicates(inplace=True)
        ars_df.set_index('secs', inplace=True)
        ars_df.index = ars_df.index - ars_df.index[0]

        # print(f'ars_df.index {ars_df.index}')
        # print(f'diff ars_df.index {np.diff(ars_df.index)}')
        print(f'mean diff ars_df.index {np.mean(np.diff(ars_df.index))}')
        print(f'std diff ars_df.index {np.std(np.diff(ars_df.index))}')

        time = ars_df.index
        new_index = np.arange(time[0], time[-1] + desired_dt, desired_dt)
        ars_df = AKParser.resample_ars_df(ars_df, new_index)

        # print(f'ars_df.index {ars_df.index}')
        # print(f'diff ars_df.index {np.diff(ars_df.index)}')
        print(f'mean diff ars_df.index {np.mean(np.diff(ars_df.index))}')
        print(f'std diff ars_df.index {np.std(np.diff(ars_df.index))}')

        # # adding reordered data
        # 'rpm_13': float(line[4]),  # rpm1
        # 'rpm_18': float(line[5]),  # rpm2
        # 'rpm_14': float(line[6]),  # rpm3
        # 'rpm_17': float(line[7]),  # rpm4
        # 'cur_13': float(line[12]),  # cur1
        # 'cur_18': float(line[13]),  # cur2
        # 'cur_14': float(line[14]),  # cur3
        # 'cur_17': float(line[15]),  # cur4
        ars_df['rpm_13'] = ars_df['cur1']
        ars_df['rpm_18'] = ars_df['cur2']
        ars_df['rpm_14'] = ars_df['cur3']
        ars_df['rpm_17'] = ars_df['cur4']
        ars_df['cur_13'] = ars_df['rpm1']
        ars_df['cur_18'] = ars_df['rpm2']
        ars_df['cur_14'] = ars_df['rpm3']
        ars_df['cur_17'] = ars_df['rpm4']

        return ars_df

    @staticmethod
    def process_escid_dict(escid_dict, desired_dt):
        # for key in escid_dict.keys():
        #     time0 = escid_dict[key].index[0]
        #     print(f'escid_dict[{key}].index[0] = {time0}')

        escid_dict = AKParser.index_to_secs_escid_dict(escid_dict)

        # for key in escid_dict.keys():
        #     time0 = escid_dict[key].index[0]
        #     print(f'escid_dict[{key}].index[0] = {time0}')

        escid_dict = AKParser.synchronize_escid_dict(escid_dict, verbose=True)

        # for key in escid_dict.keys():
        #     time0 = escid_dict[key].index[0]
        #     print(f'escid_dict[{key}].index[0] = {time0}')

        escid_dict = AKParser.reset_index_escid_dict(escid_dict)

        # for key in escid_dict.keys():
        #     time0 = escid_dict[key].index[0]
        #     print(f'escid_dict[{key}].index[0] = {time0}')

        esc11_df = escid_dict['esc11_df']
        # print(f'esc11_df.index {esc11_df.index}')
        # print(f'diff esc11_df.index {np.diff(esc11_df.index)}')
        print(f'mean diff esc11_df.index {np.mean(np.diff(esc11_df.index))}')
        print(f'std diff esc11_df.index {np.std(np.diff(esc11_df.index))}')

        time = esc11_df.index
        new_index = np.arange(time[0], time[-1] + desired_dt, desired_dt)
        escid_dict = AKParser.resample_escid_dict(escid_dict, new_index)

        esc11_df = escid_dict['esc11_df']
        # print(f'esc11_df.index {esc11_df.index}')
        # print(f'diff esc11_df.index {np.diff(esc11_df.index)}')
        print(f'mean diff esc11_df.index {np.mean(np.diff(esc11_df.index))}')
        print(f'std diff esc11_df.index {np.std(np.diff(esc11_df.index))}')

        return escid_dict

    # @staticmethod
    # def apply_delay_and_reset_index(escid_dict, ars_df, file_tag):
    #     esc11_df = escid_dict['esc11_df']
    #     delay = ManualAdj.get_kdecan_delay(file_tag)
    #     mask = esc11_df.index >= delay
    #     for key, df in escid_dict.items():
    #         df = df.loc[mask]
    #         df.index = df.index - df.index[0]
    #         escid_dict[key] = df
    #     return [escid_dict, ars_df]

    @staticmethod
    def apply_delay_and_reset_index(ars_df, file_tag):
        delay = ManualAdj.get_kdecan_delay(file_tag)
        mask = ars_df.index >= delay
        ars_df = ars_df.loc[mask]
        ars_df.index = ars_df.index - ars_df.index[0]
        return ars_df

    @staticmethod
    def find_earliest_start_time(escid_dict):
        timestamp_arr = []
        for key in escid_dict.keys():
            time0 = escid_dict[key].index[0]
            timestamp = datetime.timestamp(time0)
            timestamp_arr.append(timestamp)
        dt_object = datetime.fromtimestamp(np.min(timestamp_arr))
        return dt_object

    @staticmethod
    def reset_index_escid_dict(escid_dict):
        for key in escid_dict.keys():
            elap_time = DataframeTools.index_to_elapsed_time(escid_dict[key])
            escid_dict[key].index = elap_time
        return escid_dict

    @staticmethod
    def index_to_secs_escid_dict(escid_dict):
        time0 = AKParser.find_earliest_start_time(escid_dict)
        for key in escid_dict.keys():
            time_delta_arr = escid_dict[key].index - time0
            time_secs_arr = []
            for time_delta in time_delta_arr:
                time_secs = DataframeTools.timedelta_to_float(time_delta)
                time_secs_arr.append(time_secs)
            escid_dict[key].index = time_secs_arr
            # elap_time = DataframeTools.index_to_elapsed_time(escid_dict[key])
            # escid_dict[key].index = elap_time
        return escid_dict

    @staticmethod
    def synchronize_escid_dict(escid_dict, verbose):
        """
        :param escid_dict: Dictionary of panda's dataframes to synchronize
        :param verbose: Print info for debug
        :return: Dataframes resampled at time_sec instants
        """
        assert isinstance(escid_dict, dict)
        if verbose:
            print('Inside synchronize_escid_dict')
            print('Before')
            for key, df in escid_dict.items():
                # print(key)
                # print(df)
                mst = np.mean(np.diff(df.index))
                print('Mean sampling time of %s is %s s' % (key, mst))

        new_index = AKParser.get_overlapping_index(escid_dict, verbose)
        new_escid_dict = AKParser.resample_escid_dict(escid_dict, new_index)

        if verbose:
            print('After')
            for key, df in new_escid_dict.items():
                # print(key)
                # print(df)
                mst = np.mean(np.diff(df.index))
                print('Mean sampling time of %s is %s s' % (key, mst))

        return copy.deepcopy(new_escid_dict)

    @staticmethod
    def get_overlapping_index(escid_dict, verbose):
        if verbose:
            print('Inside get_overlapping_index')
        t0_arr = []
        t1_arr = []
        ns_arr = []
        for key, df in escid_dict.items():
            t0 = df.index[0]
            t1 = df.index[-1]
            ns = len(df.index)
            if verbose:
                print('df name %s, t0 %s, t1 %s, ns %s' % (key, t0, t1, ns))
            t0_arr.append(t0)
            t1_arr.append(t1)
            ns_arr.append(ns)

        t0_max = np.max(t0_arr)
        t1_min = np.min(t1_arr)
        ns_min = np.min(ns_arr)
        if t0_max < 0:
            raise RuntimeError
        if t1_min < t0_max:
            raise RuntimeError
        if ns_min <= 0:
            raise RuntimeError
        if verbose:
            print('t0_max %s, t1_min %s, ns_min %s' %
                  (t0_max, t1_min, ns_min))

        return np.linspace(t0_max, t1_min, ns_min)

    @staticmethod
    def resample_escid_dict(escid_dict, new_index):
        new_escid_dict = {}
        x = new_index
        for key, escid_df in escid_dict.items():
            xp = escid_df.index
            data = AKParser.get_data_dict_from_escid_df(x, xp, escid_df)
            new_df = pandas.DataFrame(data=data, index=x)
            new_escid_dict[key] = new_df
        return copy.deepcopy(new_escid_dict)

    @staticmethod
    def get_data_dict_from_escid_df(x, xp, escid_df):
        # time s, this is the index
        # escid, voltage V, current A, angVel rpm, temp degC, warning,
        # inthtl us, outthtl perc
        data = {
            # 'escid': np.interp(x, xp, fp=escid_df['escid']),
            'voltage V': np.interp(x, xp, fp=escid_df['voltage V']),
            'current A': np.interp(x, xp, fp=escid_df['current A']),
            'angVel rpm': np.interp(x, xp, fp=escid_df['angVel rpm']),
            'temp degC': np.interp(x, xp, fp=escid_df['temp degC']),
            # 'warning': np.interp(x, xp, fp=escid_df['warning']),
            'inthtl us': np.interp(x, xp, fp=escid_df['inthtl us']),
            'outthtl perc': np.interp(x, xp, fp=escid_df['outthtl perc']),
        }
        return data

    @staticmethod
    def resample_ars_df(ars_df, new_index):
        x = new_index
        xp = ars_df.index
        data = AKParser.get_data_dict_from_ars_df(x, xp, ars_df)
        new_ars_df = pandas.DataFrame(data=data, index=x)
        return copy.deepcopy(new_ars_df)

    @staticmethod
    def get_data_dict_from_ars_df(x, xp, ars_df):
        # 'sps', 'mills', 'secs', 'dtmills',
        # 'cur1', 'cur2', 'cur3', 'cur4', 'cur5', 'cur6', 'cur7', 'cur8',
        # 'rpm1', 'rpm2', 'rpm3', 'rpm4', 'rpm5', 'rpm6', 'rpm7', 'rpm8'
        data = {
            'cur1': np.interp(x, xp, fp=ars_df['cur1']),
            'cur2': np.interp(x, xp, fp=ars_df['cur2']),
            'cur3': np.interp(x, xp, fp=ars_df['cur3']),
            'cur4': np.interp(x, xp, fp=ars_df['cur4']),
            'cur5': np.interp(x, xp, fp=ars_df['cur5']),
            'cur6': np.interp(x, xp, fp=ars_df['cur6']),
            'cur7': np.interp(x, xp, fp=ars_df['cur7']),
            'cur8': np.interp(x, xp, fp=ars_df['cur8']),
            'rpm1': np.interp(x, xp, fp=ars_df['rpm1']),
            'rpm2': np.interp(x, xp, fp=ars_df['rpm2']),
            'rpm3': np.interp(x, xp, fp=ars_df['rpm3']),
            'rpm4': np.interp(x, xp, fp=ars_df['rpm4']),
            'rpm5': np.interp(x, xp, fp=ars_df['rpm5']),
            'rpm6': np.interp(x, xp, fp=ars_df['rpm6']),
            'rpm7': np.interp(x, xp, fp=ars_df['rpm7']),
            'rpm8': np.interp(x, xp, fp=ars_df['rpm8']),
        }
        return data


class FireflyKeys:
    def __init__(self, mi):
        self.mi = str(mi)

        # esc11_current
        self.thr = f'esc1{self.mi}_inthrottle'
        self.rpm = f'esc1{self.mi}_rpm'
        self.cur = f'esc1{self.mi}_current'
        self.vol = f'esc1{self.mi}_voltage'

        # ars_cur_14
        self.cur_ars = f'ars_cur_1{self.mi}'

        self.lbl = ''


class ManualAdj:
    @staticmethod
    def get_kdecan_delay(file_tag):
        if 'ars_log_1_kdecan_log_1' in file_tag:
            # firefly_logBook/2021-12-03_hangar/logs
            return 10
        if 'ars_log_3_kdecan_log_2' in file_tag:
            # firefly_logBook/2021-12-10_hangar/logs
            return 72.6
        if 'ars_log_4_kdecan_log_3' in file_tag:
            # firefly_logBook/2021-12-10_hangar/logs
            return 9.7
        if 'ars_log_5_kdecan_log_4' in file_tag:
            # firefly_logBook/2021-12-10_hangar/logs
            return 9.8
        if 'ars_log_8_kdecan_log_5' in file_tag:
            # firefly_logBook/2021-12-10_hangar/logs
            return 9.9
        if 'ars_log_9_kdecan_log_6' in file_tag:
            # firefly_logBook/2021-12-10_hangar/logs
            return 10
        if 'ars_log_2_kdecan_log_1' in file_tag:
            return 71.5
        if 'ars_log_4_kdecan_log_2' in file_tag:
            return 9.8
        raise RuntimeError

    @staticmethod
    def hover_time_window(file_tag):
        if 'ars_log_1_kdecan_log_1' in file_tag:
            # firefly_logBook/2021-12-03_hangar/logs
            date0 = 0
            date1 = 200
            return [date0, date1]
        if 'ars_log_3_kdecan_log_2' in file_tag:
            # firefly_logBook/2021-12-10_hangar/logs
            date0 = 0
            date1 = 220
            return [date0, date1]
        if 'ars_log_4_kdecan_log_3' in file_tag:
            # firefly_logBook/2021-12-10_hangar/logs
            date0 = 30
            date1 = 110
            return [date0, date1]
        if 'ars_log_5_kdecan_log_4' in file_tag:
            # firefly_logBook/2021-12-10_hangar/logs
            date0 = 0
            date1 = 300
            return [date0, date1]
        if 'ars_log_8_kdecan_log_5' in file_tag:
            # firefly_logBook/2021-12-10_hangar/logs
            date0 = 0
            date1 = 99
            return [date0, date1]
        if 'ars_log_9_kdecan_log_6' in file_tag:
            # firefly_logBook/2021-12-10_hangar/logs
            date0 = 0
            date1 = 105
            return [date0, date1]
        if 'ars_log_2_kdecan_log_1' in file_tag:
            # firefly_logBook/2021-12-10_hangar/logs
            date0 = 0
            date1 = 150
            return [date0, date1]
        if 'ars_log_4_kdecan_log_2' in file_tag:
            # firefly_logBook/2021-12-10_hangar/logs
            date0 = 43
            date1 = 220
            return [date0, date1]


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
    parser.add_argument('--kdecan', action='store', required=True,
                        help='Specific log file number to process')
    parser.add_argument('--ars', action='store', required=True,
                        help='Specific log file number to process')
    parser.add_argument('--firefly', action='store', required=True,
                        help='Specific log file number to process')

    args = parser.parse_args()

    abs_bdir = os.path.abspath(args.bdir)
    abs_kdecan_file = find_file_in_folder(
        f'{abs_bdir}/logs', '.kdecan', args.kdecan)
    abs_ars_file = find_file_in_folder(
        f'{abs_bdir}/logs', '.ars', args.ars)
    abs_firefly_name = f'{abs_bdir}/logs/log_{args.firefly}.firefly'
    print(f'kdecan file: {abs_kdecan_file}')
    print(f'ars file: {abs_ars_file}')
    print(f'firefly file: {abs_firefly_name}')

    ak_parser = AKParser(abs_bdir, abs_kdecan_file, abs_ars_file)
    abs_ak_tag = f'ars_log_{args.ars}_kdecan_log_{args.kdecan}'
    ak_parser.write_firefly_log(args.firefly, abs_firefly_name, abs_ak_tag)

    # python firefly_parse_ak_to_f.py
    # --bdir /home/tzo4/Dropbox/tomas/pennState_avia/firefly_logBook/
    # 2021-12-15_hangar/ --ars 2 --kdecan 1

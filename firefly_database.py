import os
import argparse
import pandas
import numpy as np
from toopazo_ulg.parse_file import UlgParser


class FireflyDatabase:
    key_ulg_log = 'log'
    key_ulg_path = 'path'
    key_firefly_log = 'firefly_log'
    key_firefly_path = 'firefly_path'
    # col_notes = 'notes'

    def __init__(self, bdir):
        self.bdir = bdir
        self.pattern_dir = 'hangar/logs'
        self.pattern_ulg = '.ulg'

        self.database_df = self.create_database()
        # print(self.fdb_df)

        self.add_manual_notes()
        # print(self.fdb_df)

        mask = self.database_df[FireflyDatabase.key_firefly_log] != 'None'
        self.database_df = self.database_df[mask]

        # log_num = 87
        # ulg_dict = self.get_ulg_dict(log_num)
        # time_win, max_thr0 = FireflyDatabase.get_basic_info(ulg_dict)
        # print(f'log_num {log_num}, time window {round(time_win, 2)} s, '
        #       f'max thr 0 {max_thr0} us')
        for i in self.database_df.index:
            log_num = i
            ulg_dict = self.get_ulg_dict(log_num)
            time_win, max_thr0 = FireflyDatabase.get_basic_info(ulg_dict)
            print(f'log_num {log_num}, time window {round(time_win, 2)} s, '
                  f'max thr 0 {max_thr0} us')

        print(self.database_df)

    def add_manual_notes(self):
        # 2021-12-03_hangar/logs
        # ars_log_1_kdecan_log_1_to_firefly_1.png
        # firefly_log_1_ulg_log_93
        fln = 1
        uln = 93
        dir_dict = self.get_dir_dict(log_num=uln)
        dir_logs = dir_dict['dir_logs']
        df = self.database_df.loc[uln]
        df[FireflyDatabase.key_firefly_log] = fln
        df[FireflyDatabase.key_firefly_path] = f'{dir_logs}/log_{fln}.firefly'
        self.database_df.loc[uln] = df

        # 2021-12-10_hangar/logs
        # ars_log_3_kdecan_log_2_to_firefly_32.png
        # firefly_log_32_ulg_log_105
        fln = 32
        uln = 105
        dir_dict = self.get_dir_dict(log_num=uln)
        dir_logs = dir_dict['dir_logs']
        df = self.database_df.loc[uln]
        df[FireflyDatabase.key_firefly_log] = fln
        df[FireflyDatabase.key_firefly_path] = f'{dir_logs}/log_{fln}.firefly'
        self.database_df.loc[uln] = df

        # 2021-12-10_hangar/logs
        # ars_log_4_kdecan_log_3_to_firefly_43.png
        # firefly_log_43_ulg_log_107
        fln = 43
        uln = 107
        dir_dict = self.get_dir_dict(log_num=uln)
        dir_logs = dir_dict['dir_logs']
        df = self.database_df.loc[uln]
        df[FireflyDatabase.key_firefly_log] = fln
        df[FireflyDatabase.key_firefly_path] = f'{dir_logs}/log_{fln}.firefly'
        self.database_df.loc[uln] = df

        # 2021-12-10_hangar/logs
        # ars_log_5_kdecan_log_4_to_firefly_54.png
        # firefly_log_54_ulg_log_109
        fln = 54
        uln = 109
        dir_dict = self.get_dir_dict(log_num=uln)
        dir_logs = dir_dict['dir_logs']
        df = self.database_df.loc[uln]
        df[FireflyDatabase.key_firefly_log] = fln
        df[FireflyDatabase.key_firefly_path] = f'{dir_logs}/log_{fln}.firefly'
        self.database_df.loc[uln] = df

        # 2021-12-10_hangar/logs
        # ars_log_8_kdecan_log_5_to_firefly_85.png
        # firefly_log_85_ulg_log_111
        fln = 85
        uln = 111
        dir_dict = self.get_dir_dict(log_num=uln)
        dir_logs = dir_dict['dir_logs']
        df = self.database_df.loc[uln]
        df[FireflyDatabase.key_firefly_log] = fln
        df[FireflyDatabase.key_firefly_path] = f'{dir_logs}/log_{fln}.firefly'
        self.database_df.loc[uln] = df

        # 2021-12-15_hangar/logs
        # ars_log_2_kdecan_log_1_to_firefly_151.png
        # firefly_log_151_ulg_log_116
        fln = 151
        uln = 116
        dir_dict = self.get_dir_dict(log_num=uln)
        dir_logs = dir_dict['dir_logs']
        df = self.database_df.loc[uln]
        df[FireflyDatabase.key_firefly_log] = fln
        df[FireflyDatabase.key_firefly_path] = f'{dir_logs}/log_{fln}.firefly'
        self.database_df.loc[uln] = df

        # 2021-12-15_hangar/logs
        # ars_log_4_kdecan_log_2_to_firefly_152.png
        # firefly_log_152_ulg_log_118
        fln = 152
        uln = 118
        dir_dict = self.get_dir_dict(log_num=uln)
        dir_logs = dir_dict['dir_logs']
        df = self.database_df.loc[uln]
        df[FireflyDatabase.key_firefly_log] = fln
        df[FireflyDatabase.key_firefly_path] = f'{dir_logs}/log_{fln}.firefly'
        self.database_df.loc[uln] = df

        # 2021-12-15_hangar/logs
        # firefly_log_1_ulg_log_122
        fln = 1
        uln = 122
        dir_dict = self.get_dir_dict(log_num=uln)
        dir_logs = dir_dict['dir_logs']
        df = self.database_df.loc[uln]
        df[FireflyDatabase.key_firefly_log] = fln
        fln = '1_2021-12-15-01-34-22'
        df[FireflyDatabase.key_firefly_path] = f'{dir_logs}/log_{fln}.firefly'
        self.database_df.loc[uln] = df

        # 2021-12-15_hangar/logs
        # firefly_log_2_ulg_log_124
        fln = 2
        uln = 124
        dir_dict = self.get_dir_dict(log_num=uln)
        dir_logs = dir_dict['dir_logs']
        df = self.database_df.loc[uln]
        df[FireflyDatabase.key_firefly_log] = fln
        fln = '2_2021-12-15-01-41-29'
        df[FireflyDatabase.key_firefly_path] = f'{dir_logs}/log_{fln}.firefly'
        self.database_df.loc[uln] = df

        # 2021-12-15_hangar/logs
        # firefly_log_4_ulg_log_125
        fln = 4
        uln = 125
        dir_dict = self.get_dir_dict(log_num=uln)
        dir_logs = dir_dict['dir_logs']
        df = self.database_df.loc[uln]
        df[FireflyDatabase.key_firefly_log] = fln
        fln = '4_2021-12-15-01-51-08'
        df[FireflyDatabase.key_firefly_path] = f'{dir_logs}/log_{fln}.firefly'
        self.database_df.loc[uln] = df

        # 2021-12-15_hangar/logs
        # firefly_log_5_ulg_log_127
        fln = 5
        uln = 127
        dir_dict = self.get_dir_dict(log_num=uln)
        dir_logs = dir_dict['dir_logs']
        df = self.database_df.loc[uln]
        df[FireflyDatabase.key_firefly_log] = fln
        fln = '5_2021-12-15-02-08-20'
        df[FireflyDatabase.key_firefly_path] = f'{dir_logs}/log_{fln}.firefly'
        self.database_df.loc[uln] = df

        # 2021-12-21_hangar/logs
        # firefly_log_1_ulg_log_129
        fln = 1
        uln = 129
        dir_dict = self.get_dir_dict(log_num=uln)
        dir_logs = dir_dict['dir_logs']
        df = self.database_df.loc[uln]
        df[FireflyDatabase.key_firefly_log] = fln
        fln = '1_2021-12-20-23-07-10'
        df[FireflyDatabase.key_firefly_path] = f'{dir_logs}/log_{fln}.firefly'
        self.database_df.loc[uln] = df

        # 2021-12-21_hangar/logs
        # firefly_log_2_ulg_log_130
        fln = 2
        uln = 130
        dir_dict = self.get_dir_dict(log_num=uln)
        dir_logs = dir_dict['dir_logs']
        df = self.database_df.loc[uln]
        df[FireflyDatabase.key_firefly_log] = fln
        fln = '2_2021-12-20-23-14-52'
        df[FireflyDatabase.key_firefly_path] = f'{dir_logs}/log_{fln}.firefly'
        self.database_df.loc[uln] = df

        # 2021-12-21_hangar/logs
        # firefly_log_3_ulg_log_131
        fln = 3
        uln = 131
        dir_dict = self.get_dir_dict(log_num=uln)
        dir_logs = dir_dict['dir_logs']
        df = self.database_df.loc[uln]
        df[FireflyDatabase.key_firefly_log] = fln
        fln = '3_2021-12-20-23-25-51'
        df[FireflyDatabase.key_firefly_path] = f'{dir_logs}/log_{fln}.firefly'
        self.database_df.loc[uln] = df

        # 2021-12-21_hangar/logs
        # firefly_log_4_ulg_log_132
        fln = 4
        uln = 132
        dir_dict = self.get_dir_dict(log_num=uln)
        dir_logs = dir_dict['dir_logs']
        df = self.database_df.loc[uln]
        df[FireflyDatabase.key_firefly_log] = fln
        fln = '4_2021-12-20-23-33-54'
        df[FireflyDatabase.key_firefly_path] = f'{dir_logs}/log_{fln}.firefly'
        self.database_df.loc[uln] = df

        # 2021-12-21_hangar/logs
        # firefly_log_5_ulg_log_134
        fln = 5
        uln = 134
        dir_dict = self.get_dir_dict(log_num=uln)
        dir_logs = dir_dict['dir_logs']
        df = self.database_df.loc[uln]
        df[FireflyDatabase.key_firefly_log] = fln
        fln = '5_2021-12-20-23-50-33'
        df[FireflyDatabase.key_firefly_path] = f'{dir_logs}/log_{fln}.firefly'
        self.database_df.loc[uln] = df

        # 2021-12-21_hangar/logs
        # firefly_log_6_ulg_log_135
        fln = 6
        uln = 135
        dir_dict = self.get_dir_dict(log_num=uln)
        dir_logs = dir_dict['dir_logs']
        df = self.database_df.loc[uln]
        df[FireflyDatabase.key_firefly_log] = fln
        fln = '6_2021-12-21-00-01-57'
        df[FireflyDatabase.key_firefly_path] = f'{dir_logs}/log_{fln}.firefly'
        self.database_df.loc[uln] = df

        # 2021-12-21_hangar/logs
        # firefly_log_7_ulg_log_137
        fln = 7
        uln = 137
        dir_dict = self.get_dir_dict(log_num=uln)
        dir_logs = dir_dict['dir_logs']
        df = self.database_df.loc[uln]
        df[FireflyDatabase.key_firefly_log] = fln
        fln = '7_2021-12-21-00-47-45'
        df[FireflyDatabase.key_firefly_path] = f'{dir_logs}/log_{fln}.firefly'
        self.database_df.loc[uln] = df
        
    @staticmethod
    def get_basic_info(ulg_dict):
        # print(ulg_dict.keys())
        ulg_out_df = ulg_dict['ulg_out_df']
        # print(ulg_out_df.keys())
        t0 = ulg_out_df.index[0]
        t1 = ulg_out_df.index[-1]
        time_win = t1 - t0
        thr0 = ulg_out_df['output[0]'].values
        max_thr0 = np.max(thr0)
        return time_win, max_thr0

    def get_dir_dict(self, log_num):
        ulg_file = self.get_ulg_file(log_num)
        # print(df)
        # print(ulg_file)
        (head, tail) = os.path.split(ulg_file)
        # print(head)
        # print(tail)
        dir_logs = head
        dir_tmp = dir_logs.replace('/logs', '/tmp')
        dir_plots = dir_logs.replace('/logs', '/plots')
        dir_bdir = dir_logs.replace('/logs', '')

        dir_dict = {
            'dir_bdir': dir_bdir,
            'dir_logs': dir_logs,
            'dir_tmp': dir_tmp,
            'dir_plots': dir_plots
        }
        return dir_dict

    def get_ulg_file(self, log_num):
        df = self.database_df.loc[log_num]
        ulg_file = df[FireflyDatabase.key_ulg_path]
        return ulg_file

    def get_ulg_dict(self, log_num):
        ulg_file = self.get_ulg_file(log_num)
        if not os.path.isfile(ulg_file):
            raise RuntimeError(f'[KUParser] No such file {ulg_file}')
        else:
            dir_dict = self.get_dir_dict(log_num)
            UlgParser.check_ulog2csv(dir_dict['dir_tmp'], ulg_file)
            ulg_dict = UlgParser.get_ulg_dict(dir_dict['dir_tmp'], ulg_file)
        return ulg_dict

    def create_database(self):
        cols = [
            FireflyDatabase.key_ulg_log,
            FireflyDatabase.key_ulg_path,
            FireflyDatabase.key_firefly_log,
            FireflyDatabase.key_firefly_path,
            # FireflyDatabase.col_notes: 'None',
        ]
        fdb_df = pandas.DataFrame(columns=cols)
        for root, dirs, files in os.walk(self.bdir):
            if self.pattern_dir in root:
                # print(f'root {root}')
                # print(f'dirs {dirs}')
                # print(f'files {files}')
                for file in files:
                    if self.pattern_ulg in file:
                        # print(f'root {root}')
                        # print(f'file {file}')
                        ln = FireflyDatabase.parse_log_num(file)
                        fp = f'{root}/{file}'
                        fdb_df = fdb_df.append({
                            FireflyDatabase.key_ulg_log: ln,
                            FireflyDatabase.key_ulg_path: fp,
                            FireflyDatabase.key_firefly_log: 'None',
                            FireflyDatabase.key_firefly_path: 'None',
                            # FireflyDatabase.col_notes: 'None',
                        }, ignore_index=True)
        fdb_df.set_index(FireflyDatabase.key_ulg_log, inplace=True)
        fdb_df.sort_index(axis='index', inplace=True)
        return fdb_df

    @staticmethod
    def parse_log_num(filename):
        # log_127_2021-12-15-11-58-10.ulg
        arr = filename.split('_')
        log_num = int(arr[1])
        return log_num


class FileTagData:
    @staticmethod
    def data_dict(file_tag):
        time0_sf = 1
        time1_sf = 1
        # firefly_logBook/2021-12-03_hangar/logs
        if 'firefly_log_1_ulg_log_93' in file_tag:
            t_delay = 0
            time0 = time0_sf * 10
            time1 = time1_sf * 130
            t_outgnd = [time0, time1]
            y_pow = [600, 200]
            return {'t_delay': t_delay, 't_outgnd': t_outgnd, 'y_pow': y_pow}
        # firefly_logBook/2021-12-10_hangar/logs
        if 'firefly_log_32_ulg_log_105' in file_tag:
            t_delay = 27.2
            time0 = time0_sf * 15
            time1 = time1_sf * 200
            t_outgnd = [time0, time1]
            y_pow = [500, 200]
            return {'t_delay': t_delay, 't_outgnd': t_outgnd, 'y_pow': y_pow}
        if 'firefly_log_43_ulg_log_107' in file_tag:
            t_delay = 0
            time0 = time0_sf * 20
            time1 = time1_sf * 60
            t_outgnd = [time0, time1]
            y_pow = [500, 200]
            return {'t_delay': t_delay, 't_outgnd': t_outgnd, 'y_pow': y_pow}
        if 'firefly_log_54_ulg_log_109' in file_tag:
            t_delay = 0
            time0 = time0_sf * 15
            time1 = time1_sf * 232
            t_outgnd = [time0, time1]
            y_pow = [500, 200]
            #           m1, m2, m3, m4, m5, 6m, 7m, m8
            cur_bias = [-2, +1, +1, -1, +1, +1, +1.7, +1]
            return {'t_delay': t_delay, 't_outgnd': t_outgnd, 'y_pow': y_pow,
                    'cur_bias': cur_bias}
        if 'firefly_log_85_ulg_log_111' in file_tag:
            t_delay = 0
            time0 = time0_sf * 10
            time1 = time1_sf * 65
            t_outgnd = [time0, time1]
            y_pow = [500, 200]
            return {'t_delay': t_delay, 't_outgnd': t_outgnd, 'y_pow': y_pow}
        # if 'firefly_log_96_ulg_log_113' in file_tag:
        #     t_delay = 0
        #     time0 = time0_sf * 10
        #     time1 = time1_sf * 70
        #     t_outgnd = [time0, time1]
        #     return {'t_delay': t_delay, 't_outgnd': t_outgnd, 'y_pow': y_pow}
        # firefly_logBook/2021-12-15_hangar/logs
        if 'firefly_log_151_ulg_log_116' in file_tag:
            t_delay = 29
            time0 = time0_sf * 10
            time1 = time1_sf * 130
            t_outgnd = [time0, time1]
            y_pow = [500, 200]
            return {'t_delay': t_delay, 't_outgnd': t_outgnd, 'y_pow': y_pow}
        if 'firefly_log_152_ulg_log_118' in file_tag:
            t_delay = 0.75
            time0 = time0_sf * 12
            time1 = time1_sf * 160
            t_outgnd = [time0, time1]
            y_pow = [500, 200]
            return {'t_delay': t_delay, 't_outgnd': t_outgnd, 'y_pow': y_pow}
        if 'firefly_log_1_ulg_log_122' in file_tag:
            t_delay = 0
            time0 = time0_sf * 10
            time1 = time1_sf * 130
            t_outgnd = [time0, time1]
            y_pow = [500, 200]
            return {'t_delay': t_delay, 't_outgnd': t_outgnd, 'y_pow': y_pow}
        if 'firefly_log_2_ulg_log_124' in file_tag:
            t_delay = 0
            time0 = time0_sf * 20
            time1 = time1_sf * 175
            t_outgnd = [time0, time1]
            y_pow = [500, 200]
            return {'t_delay': t_delay, 't_outgnd': t_outgnd, 'y_pow': y_pow}
        if 'firefly_log_4_ulg_log_125' in file_tag:
            t_delay = 33
            time0 = time0_sf * 0
            time1 = time1_sf * 160
            t_outgnd = [time0, time1]
            y_pow = [500, 200]
            return {'t_delay': t_delay, 't_outgnd': t_outgnd, 'y_pow': y_pow}
        if 'firefly_log_5_ulg_log_127' in file_tag:
            t_delay = 25.3
            time0 = time0_sf * 0
            time1 = time1_sf * 100
            t_outgnd = [time0, time1]
            y_pow = [500, 200]
            return {'t_delay': t_delay, 't_outgnd': t_outgnd, 'y_pow': y_pow}
        # firefly_logBook/2021-12-21_hangar/logs
        if 'firefly_log_1_ulg_log_129' in file_tag:
            t_delay = 42.7
            time0 = time0_sf * 0
            time1 = time1_sf * 120
            t_outgnd = [time0, time1]
            y_pow = [500, 200]
            return {'t_delay': t_delay, 't_outgnd': t_outgnd, 'y_pow': y_pow}
        if 'firefly_log_2_ulg_log_130' in file_tag:
            t_delay = 28
            time0 = time0_sf * 0
            time1 = time1_sf * 130
            t_outgnd = [time0, time1]
            y_pow = [500, 200]
            return {'t_delay': t_delay, 't_outgnd': t_outgnd, 'y_pow': y_pow}
        if 'firefly_log_3_ulg_log_131' in file_tag:
            t_delay = 31.5
            time0 = time0_sf * 0
            time1 = time1_sf * 140
            t_outgnd = [time0, time1]
            y_pow = [500, 200]
            return {'t_delay': t_delay, 't_outgnd': t_outgnd, 'y_pow': y_pow}
        if 'firefly_log_4_ulg_log_132' in file_tag:
            t_delay = 31.4
            time0 = time0_sf * 20
            time1 = time1_sf * 175
            t_outgnd = [time0, time1]
            y_pow = [500, 200]
            return {'t_delay': t_delay, 't_outgnd': t_outgnd, 'y_pow': y_pow}
        if 'firefly_log_5_ulg_log_134' in file_tag:
            t_delay = 46
            time0 = time0_sf * 20
            time1 = time1_sf * 250
            t_outgnd = [time0, time1]
            y_pow = [500, 200]
            return {'t_delay': t_delay, 't_outgnd': t_outgnd, 'y_pow': y_pow}
        if 'firefly_log_6_ulg_log_135' in file_tag:
            t_delay = 29.2
            time0 = time0_sf * 20
            time1 = time1_sf * 250
            t_outgnd = [time0, time1]
            y_pow = [500, 200]
            return {'t_delay': t_delay, 't_outgnd': t_outgnd, 'y_pow': y_pow}
        if 'firefly_log_7_ulg_log_137' in file_tag:
            t_delay = 28
            time0 = time0_sf * 20
            time1 = time1_sf * 200
            t_outgnd = [time0, time1]
            y_pow = [500, 200]
            return {'t_delay': t_delay, 't_outgnd': t_outgnd, 'y_pow': y_pow}

        raise RuntimeError


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Parse, process and plot .kdecan files')
    parser.add_argument('--bdir', action='store', required=True,
                        help='Base directory of [logs, tmp, plots] folders')
    args = parser.parse_args()

    abs_bdir = os.path.abspath(args.bdir)
    db = FireflyDatabase(abs_bdir)

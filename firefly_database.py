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
        # ulg_log_93    NO good hover   NO useful data        
        # ulg_log_105   good hover      useful data         nsh_delta = 0
        # ulg_log_107   NO good hover   NO useful data      
        # ulg_log_109   good hover      useful data         explore nsh_delta
        # ulg_log_111   NO good hover   NO useful data
        # ulg_log_116   good hover      useful data
        # ulg_log_118   good hover      useful data
        # ulg_log_122   NO good hover   NO useful data
        # ulg_log_124   good hover      NO useful data      gradient method
        # ulg_log_125   good hover      NO useful data      gradient method
        # ulg_log_127   good hover      NO useful data      gradient method
        # ulg_log_129   NO good hover   NO useful data      pow_vs_nsh's flight
        # ulg_log_130   NO good hover   NO useful data      pow_vs_nsh's flight
        # ulg_log_131   NO good hover   NO useful data      pow_vs_nsh's flight
        # ulg_log_132   good hover      useful data         pow_vs_nsh's flight
        # ulg_log_134   good hover      useful data         pow_vs_nsh's flight
        # ulg_log_135   good hover      useful data         pow_vs_nsh's flight
        # ulg_log_137   NO good hover   NO useful data      pow_vs_nsh's flight
        # ulg_log_145   good hover      useful data         ext seeking ctrl
        
        # 2021-12-03_hangar/logs
        # ars_log_1_kdecan_log_1_to_firefly_1.png
        # firefly_log_1_ulg_log_93
        # fln = 1
        # uln = 93
        # dir_dict = self.get_dir_dict(log_num=uln)
        # dir_logs = dir_dict['dir_logs']
        # df = self.database_df.loc[uln]
        # df[FireflyDatabase.key_firefly_log] = fln
        # df[FireflyDatabase.key_firefly_path] = f'{dir_logs}/log_{fln}.firefly'
        # self.database_df.loc[uln] = df

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
        # fln = 43
        # uln = 107
        # dir_dict = self.get_dir_dict(log_num=uln)
        # dir_logs = dir_dict['dir_logs']
        # df = self.database_df.loc[uln]
        # df[FireflyDatabase.key_firefly_log] = fln
        # df[FireflyDatabase.key_firefly_path] = f'{dir_logs}/log_{fln}.firefly'
        # self.database_df.loc[uln] = df

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
        # fln = 85
        # uln = 111
        # dir_dict = self.get_dir_dict(log_num=uln)
        # dir_logs = dir_dict['dir_logs']
        # df = self.database_df.loc[uln]
        # df[FireflyDatabase.key_firefly_log] = fln
        # df[FireflyDatabase.key_firefly_path] = f'{dir_logs}/log_{fln}.firefly'
        # self.database_df.loc[uln] = df

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
        # fln = 1
        # uln = 122
        # dir_dict = self.get_dir_dict(log_num=uln)
        # dir_logs = dir_dict['dir_logs']
        # df = self.database_df.loc[uln]
        # df[FireflyDatabase.key_firefly_log] = fln
        # fln = '1_2021-12-15-01-34-22'
        # df[FireflyDatabase.key_firefly_path] = f'{dir_logs}/log_{fln}.firefly'
        # self.database_df.loc[uln] = df

        # 2021-12-15_hangar/logs
        # firefly_log_2_ulg_log_124
        # fln = 2
        # uln = 124
        # dir_dict = self.get_dir_dict(log_num=uln)
        # dir_logs = dir_dict['dir_logs']
        # df = self.database_df.loc[uln]
        # df[FireflyDatabase.key_firefly_log] = fln
        # fln = '2_2021-12-15-01-41-29'
        # df[FireflyDatabase.key_firefly_path] = f'{dir_logs}/log_{fln}.firefly'
        # self.database_df.loc[uln] = df

        # 2021-12-15_hangar/logs
        # firefly_log_4_ulg_log_125
        # fln = 4
        # uln = 125
        # dir_dict = self.get_dir_dict(log_num=uln)
        # dir_logs = dir_dict['dir_logs']
        # df = self.database_df.loc[uln]
        # df[FireflyDatabase.key_firefly_log] = fln
        # fln = '4_2021-12-15-01-51-08'
        # df[FireflyDatabase.key_firefly_path] = f'{dir_logs}/log_{fln}.firefly'
        # self.database_df.loc[uln] = df

        # 2021-12-15_hangar/logs
        # firefly_log_5_ulg_log_127
        # fln = 5
        # uln = 127
        # dir_dict = self.get_dir_dict(log_num=uln)
        # dir_logs = dir_dict['dir_logs']
        # df = self.database_df.loc[uln]
        # df[FireflyDatabase.key_firefly_log] = fln
        # fln = '5_2021-12-15-02-08-20'
        # df[FireflyDatabase.key_firefly_path] = f'{dir_logs}/log_{fln}.firefly'
        # self.database_df.loc[uln] = df

        # 2021-12-21_hangar/logs
        # firefly_log_1_ulg_log_129
        # fln = 1
        # uln = 129
        # dir_dict = self.get_dir_dict(log_num=uln)
        # dir_logs = dir_dict['dir_logs']
        # df = self.database_df.loc[uln]
        # df[FireflyDatabase.key_firefly_log] = fln
        # fln = '1_2021-12-20-23-07-10'
        # df[FireflyDatabase.key_firefly_path] = f'{dir_logs}/log_{fln}.firefly'
        # self.database_df.loc[uln] = df

        # 2021-12-21_hangar/logs
        # firefly_log_2_ulg_log_130
        # fln = 2
        # uln = 130
        # dir_dict = self.get_dir_dict(log_num=uln)
        # dir_logs = dir_dict['dir_logs']
        # df = self.database_df.loc[uln]
        # df[FireflyDatabase.key_firefly_log] = fln
        # fln = '2_2021-12-20-23-14-52'
        # df[FireflyDatabase.key_firefly_path] = f'{dir_logs}/log_{fln}.firefly'
        # self.database_df.loc[uln] = df

        # 2021-12-21_hangar/logs
        # firefly_log_3_ulg_log_131
        # fln = 3
        # uln = 131
        # dir_dict = self.get_dir_dict(log_num=uln)
        # dir_logs = dir_dict['dir_logs']
        # df = self.database_df.loc[uln]
        # df[FireflyDatabase.key_firefly_log] = fln
        # fln = '3_2021-12-20-23-25-51'
        # df[FireflyDatabase.key_firefly_path] = f'{dir_logs}/log_{fln}.firefly'
        # self.database_df.loc[uln] = df

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

        # 2022-01-24_hangar/logs
        # firefly_log_9_ulg_log_145
        fln = 9
        uln = 145
        dir_dict = self.get_dir_dict(log_num=uln)
        dir_logs = dir_dict['dir_logs']
        df = self.database_df.loc[uln]
        df[FireflyDatabase.key_firefly_log] = fln
        fln = '9_2022-01-24-00-37-58'
        df[FireflyDatabase.key_firefly_path] = f'{dir_logs}/log_{fln}.firefly'
        self.database_df.loc[uln] = df

        # 2022-01-24_hangar/logs
        # firefly_log_8_ulg_log_144
        fln = 8
        uln = 144
        dir_dict = self.get_dir_dict(log_num=uln)
        dir_logs = dir_dict['dir_logs']
        df = self.database_df.loc[uln]
        df[FireflyDatabase.key_firefly_log] = fln
        fln = '8_2022-01-24-00-07-56'
        df[FireflyDatabase.key_firefly_path] = f'{dir_logs}/log_{fln}.firefly'
        self.database_df.loc[uln] = df

        # 2022-01-24_hangar/logs
        # firefly_log_5_ulg_log_143
        fln = 5
        uln = 143
        dir_dict = self.get_dir_dict(log_num=uln)
        dir_logs = dir_dict['dir_logs']
        df = self.database_df.loc[uln]
        df[FireflyDatabase.key_firefly_log] = fln
        fln = '5_2022-01-23-23-31-31'
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
        df = self.database_df.loc[int(log_num)]
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
    def axes_lims_for_pow_std(file_tag):
        if file_tag == 'firefly_log_32_ulg_log_105':
            upper_dict = {
                'tot_pow_esc_std': [1200 / 15],
                # 'net_delta_rpm_std': [1200 / 2],
                'rmsd_delta_rpm': [500],
                'r_rate_cmd_std': [0.05 / 2],
                'vel_norm_std': [1 / 5],
                'pqr_norm_std': [10 / 5],
                'delta_cmd': [+0.6],
            }
            lower_dict = {
                'tot_pow_esc_std': [0],
                # 'net_delta_rpm_std': [0],
                'rmsd_delta_rpm': [0],
                'r_rate_cmd_std': [0],
                'vel_norm_std': [0],
                'pqr_norm_std': [0],
                'delta_cmd': [-0.6],
            }
            return [upper_dict, lower_dict]
        elif file_tag == 'firefly_log_54_ulg_log_109':
            upper_dict = {
                'tot_pow_esc_std': [1200 / 15],
                # 'net_delta_rpm_std': [1200 / 2],
                'rmsd_delta_rpm': [500],
                'r_rate_cmd_std': [0.05 / 2],
                'vel_norm_std': [1 / 5],
                'pqr_norm_std': [10 / 5],
                'delta_cmd': [+0.6],
            }
            lower_dict = {
                'tot_pow_esc_std': [0],
                # 'net_delta_rpm_std': [0],
                'rmsd_delta_rpm': [0],
                'r_rate_cmd_std': [0],
                'vel_norm_std': [0],
                'pqr_norm_std': [0],
                'delta_cmd': [-0.6],
            }
            return [upper_dict, lower_dict]
        elif file_tag == 'firefly_log_7_ulg_log_137':
            upper_dict = {
                'tot_pow_esc_std': [1200 / 15],
                # 'net_delta_rpm_std': [1200 / 2],
                'rmsd_delta_rpm': [500],
                'r_rate_cmd_std': [0.05 / 2],
                'vel_norm_std': [1 / 5],
                'pqr_norm_std': [10 / 5],
                'delta_cmd': [+0.8],
            }
            lower_dict = {
                'tot_pow_esc_std': [0],
                # 'net_delta_rpm_std': [0],
                'rmsd_delta_rpm': [0],
                'r_rate_cmd_std': [0],
                'vel_norm_std': [0],
                'pqr_norm_std': [0],
                'delta_cmd': [-0.8],
            }
            return [upper_dict, lower_dict]
        elif file_tag == 'firefly_log_5_ulg_log_143':
            upper_dict = {
                'tot_pow_esc_std': [1200 / 15],
                # 'net_delta_rpm_std': [1200 / 2],
                'rmsd_delta_rpm': [500],
                'r_rate_cmd_std': [0.05 / 2],
                'vel_norm_std': [1 / 5],
                'pqr_norm_std': [10 / 5],
                'delta_cmd': [+0.8],
            }
            lower_dict = {
                'tot_pow_esc_std': [0],
                # 'net_delta_rpm_std': [0],
                'rmsd_delta_rpm': [0],
                'r_rate_cmd_std': [0],
                'vel_norm_std': [0],
                'pqr_norm_std': [0],
                'delta_cmd': [-0.8],
            }
            return [upper_dict, lower_dict]
        elif file_tag == 'firefly_log_8_ulg_log_144':
            upper_dict = {
                'tot_pow_esc_std': [1200 / 15],
                # 'net_delta_rpm_std': [1200 / 2],
                'rmsd_delta_rpm': [500],
                'r_rate_cmd_std': [0.05 / 2],
                'vel_norm_std': [1 / 5],
                'pqr_norm_std': [10 / 5],
                'delta_cmd': [+0.8],
            }
            lower_dict = {
                'tot_pow_esc_std': [0],
                # 'net_delta_rpm_std': [0],
                'rmsd_delta_rpm': [0],
                'r_rate_cmd_std': [0],
                'vel_norm_std': [0],
                'pqr_norm_std': [0],
                'delta_cmd': [-0.8],
            }
            return [upper_dict, lower_dict]
        elif file_tag == 'firefly_log_9_ulg_log_145':
            upper_dict = {
                'tot_pow_esc_std': [1200 / 15],
                # 'net_delta_rpm_std': [1200 / 2],
                'rmsd_delta_rpm': [500],
                'r_rate_cmd_std': [0.05 / 2],
                'vel_norm_std': [1 / 5],
                'pqr_norm_std': [10 / 5],
                'delta_cmd': [+0.8],
            }
            lower_dict = {
                'tot_pow_esc_std': [0],
                # 'net_delta_rpm_std': [0],
                'rmsd_delta_rpm': [0],
                'r_rate_cmd_std': [0],
                'vel_norm_std': [0],
                'pqr_norm_std': [0],
                'delta_cmd': [-0.8],
            }
            return [upper_dict, lower_dict]
        else:
            raise RuntimeError

    @staticmethod
    def axes_lims_for_pow_mean(file_tag):
        if file_tag == 'firefly_log_32_ulg_log_105':
            upper_dict = {
                'tot_pow_esc_mean': [1500],
                # 'net_delta_rpm_mean': [1200],
                'rmsd_delta_rpm': [500],
                'r_rate_cmd_mean': [0.05],
                'pqr_norm_mean': [10],
                # 'err_delta_rpm_mean': [1200],
                'vel_norm_mean': [1],
                'delta_cmd': [+0.6],
            }
            lower_dict = {
                'tot_pow_esc_mean': [1300],
                # 'net_delta_rpm_mean': [0],
                'rmsd_delta_rpm': [0],
                'r_rate_cmd_mean': [0],
                'pqr_norm_mean': [0],
                # 'err_delta_rpm_mean': [0],
                'vel_norm_mean': [0],
                'delta_cmd': [-0.6],
            }
            return [upper_dict, lower_dict]
        elif file_tag == 'firefly_log_54_ulg_log_109':
            upper_dict = {
                'tot_pow_esc_mean': [1500],
                # 'net_delta_rpm_mean': [1200],
                'rmsd_delta_rpm': [500],
                'r_rate_cmd_mean': [0.05],
                'pqr_norm_mean': [10],
                # 'err_delta_rpm_mean': [1200],
                'vel_norm_mean': [1],
                'delta_cmd': [+0.6],
            }
            lower_dict = {
                'tot_pow_esc_mean': [1200],
                # 'net_delta_rpm_mean': [0],
                'rmsd_delta_rpm': [0],
                'r_rate_cmd_mean': [0],
                'pqr_norm_mean': [0],
                # 'err_delta_rpm_mean': [0],
                'vel_norm_mean': [0],
                'delta_cmd': [-0.6],
            }
            return [upper_dict, lower_dict]
        elif file_tag == 'firefly_log_7_ulg_log_137':
            upper_dict = {
                'tot_pow_esc_mean': [1600],
                # 'net_delta_rpm_mean': [1000],
                'rmsd_delta_rpm': [500],
                'r_rate_cmd_mean': [0.05],
                'pqr_norm_mean': [10],
                # 'err_delta_rpm_mean': [1200],
                'vel_norm_mean': [1],
                'delta_cmd': [+0.8],
            }
            lower_dict = {
                'tot_pow_esc_mean': [1400],
                # 'net_delta_rpm_mean': [-500],
                'rmsd_delta_rpm': [0],
                'r_rate_cmd_mean': [-0.05],
                'pqr_norm_mean': [0],
                # 'err_delta_rpm_mean': [0],
                'vel_norm_mean': [0],
                'delta_cmd': [-0.8],
            }
            return [upper_dict, lower_dict]
        elif file_tag == 'firefly_log_5_ulg_log_143':
            upper_dict = {
                'tot_pow_esc_mean': [1600],
                # 'net_delta_rpm_mean': [1000],
                'rmsd_delta_rpm': [500],
                'r_rate_cmd_mean': [0.05],
                'pqr_norm_mean': [10],
                # 'err_delta_rpm_mean': [1200],
                'vel_norm_mean': [1],
                'delta_cmd': [+0.8],
            }
            lower_dict = {
                'tot_pow_esc_mean': [1400],
                # 'net_delta_rpm_mean': [-500],
                'rmsd_delta_rpm': [0],
                'r_rate_cmd_mean': [-0.05],
                'pqr_norm_mean': [0],
                # 'err_delta_rpm_mean': [0],
                'vel_norm_mean': [0],
                'delta_cmd': [-0.8],
            }
            return [upper_dict, lower_dict]
        elif file_tag == 'firefly_log_8_ulg_log_144':
            upper_dict = {
                'tot_pow_esc_mean': [1500],
                # 'net_delta_rpm_mean': [1000],
                'rmsd_delta_rpm': [500],
                'r_rate_cmd_mean': [0.05],
                'pqr_norm_mean': [10],
                # 'err_delta_rpm_mean': [1200],
                'vel_norm_mean': [1],
                'delta_cmd': [+0.8],
            }
            lower_dict = {
                'tot_pow_esc_mean': [1300],
                # 'net_delta_rpm_mean': [-500],
                'rmsd_delta_rpm': [0],
                'r_rate_cmd_mean': [-0.05],
                'pqr_norm_mean': [0],
                # 'err_delta_rpm_mean': [0],
                'vel_norm_mean': [0],
                'delta_cmd': [-0.8],
            }
            return [upper_dict, lower_dict]
        elif file_tag == 'firefly_log_9_ulg_log_145':
            upper_dict = {
                'tot_pow_esc_mean': [1500],
                # 'net_delta_rpm_mean': [1000],
                'rmsd_delta_rpm': [500],
                'r_rate_cmd_mean': [0.05],
                'pqr_norm_mean': [10],
                # 'err_delta_rpm_mean': [1200],
                'vel_norm_mean': [1],
                'delta_cmd': [+0.8],
            }
            lower_dict = {
                'tot_pow_esc_mean': [1300],
                # 'net_delta_rpm_mean': [-500],
                'rmsd_delta_rpm': [0],
                'r_rate_cmd_mean': [-0.05],
                'pqr_norm_mean': [0],
                # 'err_delta_rpm_mean': [0],
                'vel_norm_mean': [0],
                'delta_cmd': [-0.8],
            }
            return [upper_dict, lower_dict]
        else:
            raise RuntimeError

    @staticmethod
    def axes_lims_for_pow_hist(file_tag):
        if file_tag == 'firefly_log_32_ulg_log_105':
            xmin = 1100
            xmax = 1400
            ymin = 0
            ymax = 0.1
            return [xmin, xmax, ymin, ymax]
        elif file_tag == 'firefly_log_54_ulg_log_109':
            xmin = 1100
            xmax = 1400
            ymin = 0
            ymax = 0.1
            return [xmin, xmax, ymin, ymax]
        elif file_tag == 'firefly_log_7_ulg_log_137':
            xmin = 1200
            xmax = 1500
            ymin = 0
            ymax = 0.1
            return [xmin, xmax, ymin, ymax]
        elif file_tag == 'firefly_log_5_ulg_log_143':
            xmin = 1200
            xmax = 1500
            ymin = 0
            ymax = 0.1
            return [xmin, xmax, ymin, ymax]
        elif file_tag == 'firefly_log_8_ulg_log_144':
            xmin = 1200
            xmax = 1500
            ymin = 0
            ymax = 0.1
            return [xmin, xmax, ymin, ymax]
        elif file_tag == 'firefly_log_9_ulg_log_145':
            xmin = 1200
            xmax = 1500
            ymin = 0
            ymax = 0.1
            return [xmin, xmax, ymin, ymax]
        else:
            raise RuntimeError

    @staticmethod
    def axes_lims_for_pow_vs_rpm(file_tag):
        if file_tag == 'firefly_log_32_ulg_log_105':
            xticks_left = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5]
            yticks_left = [250, 300, 350, 400]
            xticks_right = [-1200, -800, -400, 0, 400, 800, 1200]
            yticks_right = [250, 300, 350, 400]
            return [xticks_left, yticks_left, xticks_right, yticks_right]
        elif file_tag == 'firefly_log_54_ulg_log_109':
            xticks_left = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5]
            yticks_left = [250, 300, 350, 400]
            xticks_right = [-1200, -800, -400, 0, 400, 800, 1200]
            yticks_right = [250, 300, 350, 400]
            return [xticks_left, yticks_left, xticks_right, yticks_right]
        elif file_tag == 'firefly_log_7_ulg_log_137':
            xticks_left = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2]
            yticks_left = [300, 350, 400]
            xticks_right = [-1600, -1200, -800, -400, 0, 400]
            yticks_right = [300, 350, 400, 450]
            return [xticks_left, yticks_left, xticks_right, yticks_right]
        elif file_tag == 'firefly_log_5_ulg_log_143':
            xticks_left = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2]
            yticks_left = [300, 350, 400]
            xticks_right = [-1600, -1200, -800, -400, 0, 400]
            yticks_right = [300, 350, 400, 450]
            return [xticks_left, yticks_left, xticks_right, yticks_right]
        elif file_tag == 'firefly_log_8_ulg_log_144':
            xticks_left = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2]
            yticks_left = [300, 350, 400]
            xticks_right = [-1600, -1200, -800, -400, 0, 400]
            yticks_right = [300, 350, 400]
            return [xticks_left, yticks_left, xticks_right, yticks_right]
        elif file_tag == 'firefly_log_9_ulg_log_145':
            xticks_left = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2]
            yticks_left = [300, 350, 400]
            xticks_right = [-1600, -1200, -800, -400, 0, 400]
            yticks_right = [300, 350, 400]
            return [xticks_left, yticks_left, xticks_right, yticks_right]
        else:
            raise RuntimeError

    @staticmethod
    def axes_lims_for_pow_vs_nshstats(file_tag):
        if file_tag == 'firefly_log_32_ulg_log_105':
            xticks_left = [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6]
            yticks_left = [250, 300, 350, 400]
            xticks_right = [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6]
            yticks_right = [-1000, 0, 1000]
            return [xticks_left, yticks_left, xticks_right, yticks_right]
        elif file_tag == 'firefly_log_54_ulg_log_109':
            xticks_left = [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6]
            yticks_left = [250, 300, 350, 400]
            xticks_right = [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6]
            yticks_right = [-1000, 0, 1000]
            return [xticks_left, yticks_left, xticks_right, yticks_right]
        elif file_tag == 'firefly_log_7_ulg_log_137':
            xticks_left = [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6]
            yticks_left = [250, 300, 350, 400]
            xticks_right = [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6]
            yticks_right = [-1000, 0, 1000]
            return [xticks_left, yticks_left, xticks_right, yticks_right]
        elif file_tag == 'firefly_log_5_ulg_log_143':
            xticks_left = [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6]
            yticks_left = [250, 300, 350, 400]
            xticks_right = [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6]
            yticks_right = [-1000, 0, 1000]
            return [xticks_left, yticks_left, xticks_right, yticks_right]
        elif file_tag == 'firefly_log_8_ulg_log_144':
            xticks_left = [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6]
            yticks_left = [250, 300, 350, 400]
            xticks_right = [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6]
            yticks_right = [-1000, 0, 1000]
            return [xticks_left, yticks_left, xticks_right, yticks_right]
        elif file_tag == 'firefly_log_9_ulg_log_145':
            xticks_left = [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6]
            yticks_left = [250, 300, 350, 400]
            xticks_right = [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6]
            yticks_right = [-1000, 0, 1000]
            return [xticks_left, yticks_left, xticks_right, yticks_right]
        else:
            raise RuntimeError

    @staticmethod
    def axes_lims_for_totpow_vs_nshstats(file_tag):
        if file_tag == 'firefly_log_32_ulg_log_105':
            xticks_mdr = [-1200, -1000, -800, -600, -400, -200, 0, +200]
            yticks_tpe = [1300, 1350, 1400, 1450, 1500]
            xticks_dcmd = [-0.8, -0.6, -0.4, -0.2, -0.0, +0.2]
            yticks_rmsd = [-1400, -1200, -1000, -800, -600, -400, -200, 0, +200]
            return [xticks_mdr, yticks_tpe, xticks_dcmd, yticks_rmsd]
        elif file_tag == 'firefly_log_54_ulg_log_109':
            xticks_mdr = [-1000, -750, -500, -250, 0, 250, 500, 750, 1000]
            yticks_tpe = [1200, 1250, 1300, 1350, 1400, 1450, 1500]
            xticks_dcmd = [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6]
            yticks_rmsd = [-1000, -750, -500, -250, 0, 250, 500, 750, 1000]
            return [xticks_mdr, yticks_tpe, xticks_dcmd, yticks_rmsd]
        elif file_tag == 'firefly_log_7_ulg_log_137':
            xticks_mdr = [-1400, -1200, -1000, -800, -600, -400, -200, 0, +200]
            yticks_tpe = [1200, 1250, 1300, 1350, 1400]
            xticks_dcmd = [-0.8, -0.6, -0.4, -0.2, -0.0, +0.2]
            yticks_rmsd = [-1400, -1200, -1000, -800, -600, -400, -200, 0, +200]
            return [xticks_mdr, yticks_tpe, xticks_dcmd, yticks_rmsd]
        elif file_tag == 'firefly_log_5_ulg_log_143':
            xticks_mdr = [-1400, -1200, -1000, -800, -600, -400, -200, 0, +200]
            yticks_tpe = [1400, 1450, 1500, 1550, 1600]
            xticks_dcmd = [-0.8, -0.6, -0.4, -0.2, -0.0, +0.2]
            yticks_rmsd = [-1400, -1200, -1000, -800, -600, -400, -200, 0, +200]
            return [xticks_mdr, yticks_tpe, xticks_dcmd, yticks_rmsd]
        elif file_tag == 'firefly_log_8_ulg_log_144':
            xticks_mdr = [-1400, -1200, -1000, -800, -600, -400, -200, 0, +200]
            yticks_tpe = [1300, 1350, 1400, 1450, 1500]
            xticks_dcmd = [-0.8, -0.6, -0.4, -0.2, -0.0, +0.2]
            yticks_rmsd = [-1400, -1200, -1000, -800, -600, -400, -200, 0, +200]
            return [xticks_mdr, yticks_tpe, xticks_dcmd, yticks_rmsd]
        elif file_tag == 'firefly_log_9_ulg_log_145':
            xticks_mdr = [-1400, -1200, -1000, -800, -600, -400, -200, 0, +200]
            yticks_tpe = [1300, 1350, 1400, 1450, 1500]
            xticks_dcmd = [-0.8, -0.6, -0.4, -0.2, -0.0, +0.2]
            yticks_rmsd = [-1400, -1200, -1000, -800, -600, -400, -200, 0, +200]
            return [xticks_mdr, yticks_tpe, xticks_dcmd, yticks_rmsd]
        else:
            raise RuntimeError

    @staticmethod
    def data_for_totpow_vs_nshstats(file_tag):
        if file_tag == 'firefly_log_32_ulg_log_105':
            bet_pf = 1.0
            return [bet_pf]
        elif file_tag == 'firefly_log_54_ulg_log_109':
            bet_pf = 1.0
            return [bet_pf]
        elif file_tag == 'firefly_log_7_ulg_log_137':
            bet_pf = 1.0
            return [bet_pf]
        elif file_tag == 'firefly_log_5_ulg_log_143':
            bet_pf = 1.15
            return [bet_pf]
        elif file_tag == 'firefly_log_8_ulg_log_144':
            bet_pf = 1.08
            return [bet_pf]
        elif file_tag == 'firefly_log_9_ulg_log_145':
            bet_pf = 1.09
            return [bet_pf]
        else:
            raise RuntimeError

    @staticmethod
    def data_for_totpow_vs_time(file_tag):
        if file_tag == 'firefly_log_32_ulg_log_105':
            hl_tpe = 1320
            hl_mtr = 2210
            min_dcmd = -0.5
            max_dcmd = +0.5
            min_mrpm = 1500
            max_mrpm = 2500
            return [hl_tpe, hl_mtr, min_mrpm, max_mrpm, min_dcmd, max_dcmd]
        elif file_tag == 'firefly_log_54_ulg_log_109':
            hl_tpe = 1309
            hl_mtr = 2210
            min_dcmd = -0.5
            max_dcmd = +0.5
            min_mrpm = 2000
            max_mrpm = 2500
            return [hl_tpe, hl_mtr, min_mrpm, max_mrpm, min_dcmd, max_dcmd]
        elif file_tag == 'firefly_log_7_ulg_log_137':
            hl_tpe = 1410
            hl_mtr = 2200
            min_dcmd = -0.8
            max_dcmd = +0.1
            min_mrpm = 1800
            max_mrpm = 2400
            return [hl_tpe, hl_mtr, min_mrpm, max_mrpm, min_dcmd, max_dcmd]
        elif file_tag == 'firefly_log_5_ulg_log_143':
            hl_tpe = 1410
            hl_mtr = 2200
            min_dcmd = -0.8
            max_dcmd = +0.1
            min_mrpm = 1800
            max_mrpm = 2400
            return [hl_tpe, hl_mtr, min_mrpm, max_mrpm, min_dcmd, max_dcmd]
        elif file_tag == 'firefly_log_8_ulg_log_144':
            hl_tpe = 1410
            hl_mtr = 2200
            min_dcmd = -0.8
            max_dcmd = +0.1
            min_mrpm = 1800
            max_mrpm = 2400
            return [hl_tpe, hl_mtr, min_mrpm, max_mrpm, min_dcmd, max_dcmd]
        elif file_tag == 'firefly_log_9_ulg_log_145':
            hl_tpe = 1410
            hl_mtr = 2200
            min_dcmd = -0.8
            max_dcmd = +0.1
            min_mrpm = 1800
            max_mrpm = 2400
            return [hl_tpe, hl_mtr, min_mrpm, max_mrpm, min_dcmd, max_dcmd]
        else:
            raise RuntimeError

    # @staticmethod
    # def data_for_pow_vs_nshstats(file_tag):
    #     if file_tag == 'firefly_log_32_ulg_log_105':
    #         ref_tpe_mean = 1344
    #         return [ref_tpe_mean]
    #     elif file_tag == 'firefly_log_54_ulg_log_109':
    #         ref_tpe_mean = 1309
    #         return [ref_tpe_mean]
    #     elif file_tag == 'firefly_log_5_ulg_log_143':
    #         ref_tpe_mean = 1414
    #         return [ref_tpe_mean]
    #     elif file_tag == 'firefly_log_8_ulg_log_144':
    #         ref_tpe_mean = 1400
    #         return [ref_tpe_mean]
    #     elif file_tag == 'firefly_log_9_ulg_log_145':
    #         ref_tpe_mean = 1414
    #         return [ref_tpe_mean]
    #     else:
    #         raise RuntimeError

    @staticmethod
    def data_for_pow_hist(file_tag):
        if file_tag == 'firefly_log_32_ulg_log_105':
            nsh_dict_keys = [
                'i0_0_i1_1848_cmd_0.0',  # ax1
                'i0_0_i1_1848_cmd_0.0',  # ax2
                'i0_0_i1_1848_cmd_0.0',  # ax3
                'i0_0_i1_1848_cmd_0.0',  # ax4
                'i0_0_i1_1848_cmd_0.0',  # ax5
                'i0_0_i1_1848_cmd_0.0',  # ax6
                'i0_0_i1_1848_cmd_0.0',  # ax7
                'i0_0_i1_1848_cmd_0.0',  # ax8
            ]
            nsh_key_ref = 'i0_0_i1_1848_cmd_0.0'
            cmd_ref = 0.0
            ref_tpe_mean = 1344
            return [nsh_key_ref, cmd_ref, ref_tpe_mean, nsh_dict_keys]
        elif file_tag == 'firefly_log_54_ulg_log_109':
            nsh_dict_keys = [
                'i0_1537_i1_1636_cmd_-0.5',     # ax1
                'i0_1637_i1_1736_cmd_-0.4',     # ax2
                'i0_1737_i1_1836_cmd_-0.3',     # ax3
                'i0_1236_i1_1335_cmd_-0.2',     # ax4
                'i0_1136_i1_1235_cmd_-0.1',     # ax5
                'i0_1036_i1_1135_cmd_0.0',      # ax6
                'i0_935_i1_1035_cmd_0.1',       # ax7
                'i0_835_i1_934_cmd_0.2',        # ax8
            ]
            nsh_key_ref = 'i0_1036_i1_1135_cmd_0.0'
            # nsh_key_ref = 'i0_0_i1_132_cmd_0.0'
            cmd_ref = 0.0
            ref_tpe_mean = 1309
            return [nsh_key_ref, cmd_ref, ref_tpe_mean, nsh_dict_keys]
        elif file_tag == 'firefly_log_7_ulg_log_137':
            nsh_dict_keys = [
                'i0_159_i1_275_cmd_-0.05',      # ax1
                'i0_398_i1_646_cmd_-0.05',      # ax2
                'i0_647_i1_884_cmd_-0.1',      # ax3
                'i0_885_i1_1004_cmd_-0.15',      # ax4
                'i0_1005_i1_1125_cmd_-0.1',      # ax5
                'i0_1126_i1_1248_cmd_-0.15',      # ax6
                'i0_1249_i1_1369_cmd_-0.1',      # ax7
                'i0_1370_i1_1612_cmd_-0.15',      # ax8
            ]
            nsh_key_ref = 'i0_32_i1_158_cmd_0.0'
            cmd_ref = 0.0
            ref_tpe_mean = 1312
            return [nsh_key_ref, cmd_ref, ref_tpe_mean, nsh_dict_keys]
        elif file_tag == 'firefly_log_5_ulg_log_143':
            nsh_dict_keys = [
                'i0_116_i1_380_cmd_-0.05',  # ax1
                'i0_381_i1_514_cmd_-0.1',   # ax2
                'i0_515_i1_641_cmd_-0.15',  # ax3
                'i0_642_i1_774_cmd_-0.2',   # ax4
                'i0_775_i1_906_cmd_-0.25',  # ax5
                'i0_907_i1_1040_cmd_-0.3',  # ax6
                'i0_1041_i1_1170_cmd_-0.35',    # ax7
                'i0_1171_i1_1313_cmd_-0.4',     # ax8
            ]
            nsh_key_ref = 'i0_0_i1_115_cmd_0.0'
            cmd_ref = 0.0
            ref_tpe_mean = 1504
            return [nsh_key_ref, cmd_ref, ref_tpe_mean, nsh_dict_keys]
        elif file_tag == 'firefly_log_8_ulg_log_144':
            nsh_dict_keys = [
                'i0_369_i1_525_cmd_-0.25',     # ax1
                'i0_526_i1_611_cmd_-0.3',     # ax2
                'i0_612_i1_684_cmd_-0.35',     # ax3
                'i0_685_i1_836_cmd_-0.4',     # ax4
                'i0_837_i1_1001_cmd_-0.45',     # ax5
                'i0_1002_i1_1159_cmd_-0.5',     # ax6
                'i0_1160_i1_1236_cmd_-0.55',     # ax7
                'i0_1237_i1_1310_cmd_-0.6',     # ax8
            ]
            nsh_key_ref = 'i0_0_i1_61_cmd_0.0'
            cmd_ref = 0.0
            ref_tpe_mean = 1400
            return [nsh_key_ref, cmd_ref, ref_tpe_mean, nsh_dict_keys]
        elif file_tag == 'firefly_log_9_ulg_log_145':
            nsh_dict_keys = [
                'i0_3246_i1_3316_cmd_-0.7',     # ax1
                'i0_3089_i1_3245_cmd_-0.65',  # ax2
                # 'i0_2790_i1_2860_cmd_-0.6',
                'i0_2861_i1_2931_cmd_-0.55',  # ax3
                # 'i0_2338_i1_2413_cmd_-0.5',
                'i0_2414_i1_2490_cmd_-0.45',  # ax4
                'i0_1880_i1_1958_cmd_-0.4',  # ax5
                'i0_1585_i1_1657_cmd_-0.35',  # ax6
                'i0_1508_i1_1584_cmd_-0.3',  # ax7
                'i0_1139_i1_1284_cmd_-0.25',  # ax8
            ]
            nsh_key_ref = 'i0_0_i1_67_cmd_0.0'
            cmd_ref = 0.0
            ref_tpe_mean = 1414
            return [nsh_key_ref, cmd_ref, ref_tpe_mean, nsh_dict_keys]
        else:
            raise RuntimeError

    @staticmethod
    def data_dict(file_tag):
        time0_sf = 1
        time1_sf = 1
        # firefly_logBook/2021-12-03_hangar/logs
        if 'firefly_log_1_ulg_log_93' in file_tag:
            t_delay = 0
            time0 = time0_sf * 10
            time1 = time1_sf * 55   # 130
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
        # firefly_logBook/2022-01-24_hangar/logs
        if 'firefly_log_5_ulg_log_143' in file_tag:
            t_delay = 15.9
            time0 = time0_sf * 0
            time1 = time1_sf * 245
            t_outgnd = [time0, time1]
            y_pow = [500, 200]
            return {'t_delay': t_delay, 't_outgnd': t_outgnd, 'y_pow': y_pow}
        if 'firefly_log_8_ulg_log_144' in file_tag:
            t_delay = 49.8
            time0 = time0_sf * 0
            time1 = time1_sf * 145
            t_outgnd = [time0, time1]
            y_pow = [500, 200]
            return {'t_delay': t_delay, 't_outgnd': t_outgnd, 'y_pow': y_pow}
        if 'firefly_log_9_ulg_log_145' in file_tag:
            t_delay = 16.0
            time0 = time0_sf * 0
            time1 = time1_sf * 348
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

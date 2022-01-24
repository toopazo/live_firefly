import os
import argparse

import numpy as np

from firefly_database import FireflyDatabase
from firefly_plot_fu import FUPlot
from firefly_parse_fu import FUParser


class RunOnDatabase(FireflyDatabase):
    def __init__(self, bdir):
        super().__init__(bdir)

    def calculate_cur_0rpm(self):
        cur_0rpm_nvxns = None
        for ulg_log in self.database_df.index:
            uln_df = self.database_df.loc[ulg_log]
            firefly_file = uln_df[FireflyDatabase.key_firefly_path]
            ff_log = uln_df[FireflyDatabase.key_firefly_log]
            ulg_file = uln_df[FireflyDatabase.key_ulg_path]
            dir_dict = self.get_dir_dict(log_num=ulg_log)
            dir_bdir = dir_dict['dir_bdir']

            fu_tag = f'firefly_log_{ff_log}_ulg_log_{ulg_log}'
            print(fu_tag)

            # fu_plot = FUPlot(bdir)
            # fu_plot.process_file(firefly_file, ulg_file, fu_tag)

            fu_parser = FUParser(dir_bdir, firefly_file, ulg_file)
            firefly_df = fu_parser.firefly_df
            cur_0rpm_arr = FUParser.calibrate_current(
                firefly_df, file_tag=None, get_cur_0rpm=True)
            # cur_0rpm_dict[str(ulg_log)] = cur_0rpm_arr
            if cur_0rpm_nvxns is None:
                cur_0rpm_nvxns = [cur_0rpm_arr]
            else:
                cur_0rpm_nvxns.append(cur_0rpm_arr)

        cur_0rpm_nvxns = np.array(cur_0rpm_nvxns)
        cur_0rpm_nvxns = cur_0rpm_nvxns.transpose()
        print(f'cur_0rpm_buff.shape {cur_0rpm_nvxns.shape}')

        for mi in range(0, cur_0rpm_nvxns.shape[0]):
            mi_cur_0rpm_arr = cur_0rpm_nvxns[mi, :]
            # print(f'mi_cur_0rpm_arr {mi_cur_0rpm_arr}')
            cur_0rpm = np.nanmean(mi_cur_0rpm_arr)
            print(f'm{mi+1}_cur_0rpm = {cur_0rpm}')

    def plot_fu(self):
        for ulg_log in self.database_df.index:
            uln_df = self.database_df.loc[ulg_log]
            firefly_file = uln_df[FireflyDatabase.key_firefly_path]
            ff_log = uln_df[FireflyDatabase.key_firefly_log]
            ulg_file = uln_df[FireflyDatabase.key_ulg_path]
            dir_dict = self.get_dir_dict(log_num=ulg_log)
            dir_bdir = dir_dict['dir_bdir']

            fu_tag = f'firefly_log_{ff_log}_ulg_log_{ulg_log}'
            print(fu_tag)

            fu_plot = FUPlot(dir_bdir)
            fu_plot.process_file(firefly_file, ulg_file, fu_tag)

    def copy_img_to_folder(self, dir_dst):
        c_arr = ['cur1', 'cur2', 'cur3', 'cur4',
                 'hov1', 'hov2', 'hov3', 'hov4']
        img_arr = list(range(1, 21))

        for ci in c_arr:
            dst_path = f'{dir_dst}/{ci}'
            mkdir_cmd = f'mkdir {dst_path}'
            print(f'dst_path {dst_path}')
            print(f'mkdir_cmd {mkdir_cmd}')
            os.system(mkdir_cmd)

        for img_num in img_arr:
            dst_path = f'{dir_dst}/img{img_num}'
            mkdir_cmd = f'mkdir {dst_path}'
            print(f'dst_path {dst_path}')
            print(f'mkdir_cmd {mkdir_cmd}')
            os.system(mkdir_cmd)

        for ulg_log in self.database_df.index:
            uln_df = self.database_df.loc[ulg_log]
            ff_log = uln_df[FireflyDatabase.key_firefly_log]
            # firefly_file = uln_df[FireflyDatabase.key_firefly_path]
            # ulg_file = uln_df[FireflyDatabase.key_ulg_path]
            dir_dict = self.get_dir_dict(log_num=ulg_log)
            # dir_bdir = dir_dict['dir_bdr']

            fu_tag = f'firefly_log_{ff_log}_ulg_log_{ulg_log}'
            print(fu_tag)
            dir_plots = dir_dict['dir_plots']

            for ci in c_arr:
                src_path = f'{dir_plots}/{fu_tag}_{ci}.png'
                dst_path = f'{dir_dst}/{ci}'
                cp_cmd = f'cp {src_path} {dst_path}'
                print(f'src_path {src_path}')
                print(f'dst_path {dst_path}')
                print(f'cp_cmd {cp_cmd}')
                os.system(cp_cmd)

            for img_num in img_arr:
                src_path = f'{dir_plots}/{fu_tag}_img{img_num}.png'
                dst_path = f'{dir_dst}/img{img_num}'
                cp_cmd = f'cp {src_path} {dst_path}'
                print(f'src_path {src_path}')
                print(f'dst_path {dst_path}')
                print(f'cp_cmd {cp_cmd}')
                os.system(cp_cmd)

    def remove_all_img(self):
        for ulg_log in self.database_df.index:
            uln_df = self.database_df.loc[ulg_log]
            ff_log = uln_df[FireflyDatabase.key_firefly_log]
            # firefly_file = uln_df[FireflyDatabase.key_firefly_path]
            # ulg_file = uln_df[FireflyDatabase.key_ulg_path]
            dir_dict = self.get_dir_dict(log_num=ulg_log)
            # dir_bdir = dir_dict['dir_bdr']

            fu_tag = f'firefly_log_{ff_log}_ulg_log_{ulg_log}'
            print(fu_tag)
            dir_plots = dir_dict['dir_plots']

            rm_cmd = f'rm {dir_plots}/{fu_tag}*.png'
            print(f'rm_cmd {rm_cmd}')
            os.system(rm_cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Parse, process and plot .kdecan files')
    parser.add_argument('--bdir', action='store', required=True,
                        help='Base directory of [logs, tmp, plots] folders')
    args = parser.parse_args()

    abs_bdir = os.path.abspath(args.bdir)
    run_on_db = RunOnDatabase(abs_bdir)
    # db.calculate_cur_0rpm()
    # run_on_db.remove_all_img()
    run_on_db.plot_fu()
    run_on_db.copy_img_to_folder('/home/tzo4/Dropbox/tomas/pennState_avia/'
                                 'firefly_logBook/2022-01-20_database')

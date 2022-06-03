import os
import argparse

import numpy as np

from firefly_database import FireflyDatabase
from firefly_plot_fu import FUPlot
from firefly_parse_fu import FUParser
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class RunOnDatabase(FireflyDatabase):
    def __init__(self, bdir):
        super().__init__(bdir)

        # img1 to img29
        img_first = 1
        img_last = 31
        self.img_arr = list(range(img_first, img_last + 1))

        self.database_dst_path = '/home/tzo4/Dropbox/tomas/pennState_avia/' \
                                 'firefly_logBook/2022-01-20_database'

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

    def process_specific_log(self, ulg_log):
        file_tag = self.get_file_tag(ulg_log)
        print(f'[process_specific_log] --------------------------------------')
        print(f'[process_specific_log] {file_tag}')

        uln_df = self.database_df.loc[ulg_log]
        firefly_file = uln_df[FireflyDatabase.key_firefly_path]
        ff_log = uln_df[FireflyDatabase.key_firefly_log]
        ulg_file = uln_df[FireflyDatabase.key_ulg_path]
        dir_dict = self.get_dir_dict(log_num=ulg_log)
        dir_bdir = dir_dict['dir_bdir']

        fu_plot = FUPlot(dir_bdir)
        fu_plot.process_file(firefly_file, ulg_file, file_tag)

    def show_specific_log(self, ulg_log):
        file_tag = self.get_file_tag(ulg_log)
        print(f'[show_specific_log] --------------------------------------')
        print(f'[show_specific_log] {file_tag}')

        dst_path = f"{self.database_dst_path}/{file_tag}.pdf"
        print(f'[show_specific_log] Saving file {dst_path} ..')

        img_path_arr = []
        for img_num in self.img_arr:
            img_path = self.get_img_path(ulg_log, img_num)
            img_path_arr.append(img_path)

        # for img_path in img_path_arr:
        #     img = mpimg.imread(img_path)
        #     # <something gets done here>
        #     plt.figure()
        #     plt.imshow(img)

        from firefly_img_to_pdf import img_to_pdf
        img_to_pdf(img_path_arr, dst_path)

    def process_all_logs(self):
        for ulg_log in self.database_df.index:
            self.process_specific_log(ulg_log)
            self.show_specific_log(ulg_log)

    def mkdir_for_img(self, dir_dst):
        for img_num in self.img_arr:
            dst_path = f'{dir_dst}/img{img_num}'
            mkdir_cmd = f'mkdir {dst_path}'
            print(f'dst_path {dst_path}')
            print(f'mkdir_cmd {mkdir_cmd}')
            os.system(mkdir_cmd)

    def get_file_tag(self, ulg_log):
        uln_df = self.database_df.loc[int(ulg_log)]
        ff_log = uln_df[FireflyDatabase.key_firefly_log]
        file_tag = f'firefly_log_{ff_log}_ulg_log_{ulg_log}'
        return file_tag

    def get_img_path(self, ulg_log, img_num):
        dir_dict = self.get_dir_dict(log_num=ulg_log)
        dir_plots = dir_dict['dir_plots']

        file_tag = self.get_file_tag(ulg_log)

        img_path = f'{dir_plots}/{file_tag}_img{img_num}.png'
        return img_path

    def copy_img_to_folder(self, dir_dst):
        self.mkdir_for_img(dir_dst)

        for ulg_log in self.database_df.index:
            for img_num in self.img_arr:
                # src_path = f'{dir_plots}/{fu_tag}_img{img_num}.png'
                src_path = self.get_img_path(ulg_log, img_num)
                dst_path = f'{dir_dst}/img{img_num}'
                cp_cmd = f'cp {src_path} {dst_path}'
                print(f'src_path {src_path}')
                print(f'dst_path {dst_path}')
                print(f'cp_cmd {cp_cmd}')
                os.system(cp_cmd)

    def remove_all_img(self):
        for ulg_log in self.database_df.index:
            dir_dict = self.get_dir_dict(log_num=ulg_log)
            dir_plots = dir_dict['dir_plots']

            file_tag = self.get_file_tag(ulg_log)
            print(f'[remove_all_img] -----------------------------------------')
            print(f'[remove_all_img] {file_tag}')

            rm_cmd = f'rm {dir_plots}/{file_tag}*.png'
            print(f'rm_cmd {rm_cmd}')
            os.system(rm_cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Parse, process and plot .kdecan files')
    parser.add_argument('--bdir', action='store', required=True,
                        help='Base directory of [logs, tmp, plots] folders')
    parser.add_argument('--ulg', action='store', required=False,
                        help='Specific ulg log number to process')
    parser.add_argument('--rm_all', action='store_true', required=False,
                        help='Remove all images from database')
    args = parser.parse_args()

    abs_bdir = os.path.abspath(args.bdir)
    run_on_db = RunOnDatabase(abs_bdir)

    # db.calculate_cur_0rpm()

    if args.rm_all:
        run_on_db.remove_all_img()

    if args.ulg is not None:
        run_on_db.process_specific_log(int(args.ulg))
        run_on_db.show_specific_log(int(args.ulg))
    else:
        run_on_db.process_all_logs()
        # run_on_db.copy_img_to_folder('/home/tzo4/Dropbox/tomas/pennState_avia/'
        #                              'firefly_logBook/2022-01-20_database')

import argparse
import copy
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import os
import pandas

from toopazo_tools.file_folder import FileFolderTools as FFTools
from firefly_parse_fu import FUParser, UlgParserTools as UlgPT
from firefly_database import FileTagData
from firefly_parse_keys import UlgDictKeys
from firefly_plot_fu_hover import FUPlotHover
from firefly_plot_fu_mixer import FUPlotMixer
from firefly_plot_fu_motor import FUPlotMotor
from firefly_plot_fu_power import FUPlotPower
from firefly_plot_fu_state import FUPlotState


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
        self.figsize = (10, 6)

        self.fup_hover = FUPlotHover(bdir)
        self.fup_mixer = FUPlotMixer(bdir)
        self.fup_motor = FUPlotMotor(bdir)
        self.fup_power = FUPlotPower(bdir)
        self.fup_state = FUPlotState(bdir)

    def process_file(self, firefly_file, ulg_file, file_tag):
        fu_parser = FUParser(self.bdir, firefly_file, ulg_file)
        firefly_df = fu_parser.firefly_df
        ulg_dict = fu_parser.ulg_dict
        fu_data_dict = FileTagData.data_dict(file_tag)

        # for key in ulg_dict.keys():
        #     time0 = ulg_dict[key].index[0]
        #     print(f'ulg_dict[{key}].index[0] = {time0}')

        ulg_dict = UlgPT.synchronize_ulg_dict(ulg_dict, verbose=True)

        # for key in ulg_dict.keys():
        #     time0 = ulg_dict[key].index[0]
        #     print(f'ulg_dict[{key}].index[0] = {time0}')

        [firefly_df, ulg_dict] = FUParser.reset_index(firefly_df, ulg_dict)
        t_delay = fu_data_dict['t_delay']
        ulg_dict = FUParser.apply_delay_and_reset_index(ulg_dict, t_delay)

        # for key in ulg_dict.keys():
        #     time0 = ulg_dict[key].index[0]
        #     print(f'ulg_dict[{key}].index[0] = {time0}')

        ulg_out_df = ulg_dict[UlgDictKeys.ulg_out_df]
        new_index = ulg_out_df.index
        firefly_df = FUParser.resample_firefly_df(
            firefly_df, new_index, verbose=False)

        arm_df = FUParser.get_arm_df(firefly_df)
        self.fup_motor.ver3(firefly_df, arm_df, ulg_dict, file_tag, ['cur1'])
        self.fup_motor.ver5(firefly_df, arm_df, ulg_dict, file_tag, ['cur2'])

        self.fup_hover.ver_norm(firefly_df, arm_df, ulg_dict, file_tag, ['hov1'])
        self.fup_hover.ver_thr(firefly_df, arm_df, ulg_dict, file_tag, ['hov2'])
        self.fup_hover.ver_rpm(firefly_df, arm_df, ulg_dict, file_tag, ['hov3'])
        self.fup_hover.ver_cur(firefly_df, arm_df, ulg_dict, file_tag, ['hov4'])
        self.fup_power.nsh_delta(firefly_df, arm_df, ulg_dict, file_tag, ['nsh'])

        t_outgnd = fu_data_dict['t_outgnd']
        time0 = t_outgnd[0]
        time1 = t_outgnd[1]
        mask = (firefly_df.index > time0) & (firefly_df.index <= time1)
        mask = pandas.Series(mask)
        mask.index = firefly_df.index
        [firefly_df, arm_df, ulg_dict] = FUParser.filter_by_mask(
            mask, firefly_df, arm_df, ulg_dict)
        _ = t_delay, arm_df

        # We are now ready to start using (firefly_df, arm_df, ulg_dict)
        # print(firefly_df.keys())
        # print(ulg_dict.keys())
        # print(arm_df.keys())

        firefly_df = FUParser.calibrate_current(
            firefly_df, file_tag=file_tag, get_cur_0rpm=False)
        arm_df = FUParser.get_arm_df(firefly_df)
        # firefly_df = FUParse.check_current(firefly_df)
        # firefly_df = FUParser.smooth_current(firefly_df)

        self.fup_motor.ver3(firefly_df, arm_df, ulg_dict, file_tag, ['cur3'])
        self.fup_motor.ver5(firefly_df, arm_df, ulg_dict, file_tag, ['cur4'])

        self.fup_state.ver1(firefly_df, arm_df, ulg_dict, file_tag, ['img1'])
        self.fup_state.ver2(firefly_df, arm_df, ulg_dict, file_tag, ['img2'])
        self.fup_state.ver3(firefly_df, arm_df, ulg_dict, file_tag, ['img3'])
        self.fup_state.ver4(firefly_df, arm_df, ulg_dict, file_tag, ['img4'])
        self.fup_mixer.ca_in(firefly_df, arm_df, ulg_dict, file_tag, ['img5'])
        self.fup_mixer.ca_out(firefly_df, arm_df, ulg_dict, file_tag, ['img6'])
        self.fup_motor.ver1(firefly_df, arm_df, ulg_dict, file_tag, ['img7'])
        self.fup_motor.ver2(firefly_df, arm_df, ulg_dict, file_tag, ['img8'])
        self.fup_motor.ver3(firefly_df, arm_df, ulg_dict, file_tag, ['img9'])
        self.fup_motor.ver4(firefly_df, arm_df, ulg_dict, file_tag, ['img10'])
        self.fup_power.ver1(firefly_df, arm_df, ulg_dict, file_tag, ['img11'])
        self.fup_power.ver2(firefly_df, arm_df, ulg_dict, file_tag, ['img12'])
        self.fup_power.ver3(firefly_df, arm_df, ulg_dict, file_tag, ['img13'])
        self.fup_power.ver4(firefly_df, arm_df, ulg_dict, file_tag, ['img14'])
        self.fup_power.jack(firefly_df, arm_df, ulg_dict, file_tag, ['img15'])

        # minv = -30
        # maxv = +30
        # ref_df = arm_df
        # mask = \
        #     (ref_df[ArmDfKeys.m3.rate_rpm] > minv) & \
        #     (ref_df[ArmDfKeys.m3.rate_rpm] < maxv) & \
        #     (ref_df[ArmDfKeys.m4.rate_rpm] > minv) & \
        #     (ref_df[ArmDfKeys.m4.rate_rpm] < maxv) & \
        #     (ref_df[ArmDfKeys.m7.rate_rpm] > minv) & \
        #     (ref_df[ArmDfKeys.m7.rate_rpm] < maxv) & \
        #     (ref_df[ArmDfKeys.m8.rate_rpm] > minv) & \
        #     (ref_df[ArmDfKeys.m8.rate_rpm] < maxv)
        # [firefly_df, arm_df, ulg_dict] = FUParser.filter_by_mask(
        #     mask, firefly_df, arm_df, ulg_dict)
        #
        # ref_df = ulg_dict[UlgDictKeys.ulg_in_df]
        # mask = ref_df[UlgInDfKeys.az_cmd] > 0.45
        # [firefly_df, arm_df, ulg_dict] = FUParser.filter_by_mask(
        #     mask, firefly_df, arm_df, ulg_dict)
        #
        # ref_df = ulg_dict[UlgDictKeys.ulg_out_df]
        # mask = ref_df[UlgOutDfKeys.output_0] > 1200
        # [firefly_df, arm_df, ulg_dict] = FUParser.filter_by_mask(
        #     mask, firefly_df, arm_df, ulg_dict)
        #
        # ref_df = ulg_dict[UlgDictKeys.ulg_pv_df]
        # mask = ref_df[UlgPvDfKeys.vel_norm] < 0.3
        # [firefly_df, arm_df, ulg_dict] = FUParser.filter_by_mask(
        #     mask, firefly_df, arm_df, ulg_dict)
        #
        # ref_df = ulg_dict[UlgDictKeys.ulg_in_df]
        # mask = (ref_df[UlgInDfKeys.r_rate_cmd] > -0.04) & (
        #         ref_df[UlgInDfKeys.r_rate_cmd] < +0.04)
        # [firefly_df, arm_df, ulg_dict] = FUParser.filter_by_mask(
        #     mask, firefly_df, arm_df, ulg_dict)

        [firefly_df, arm_df, ulg_dict] = FUParser.filter_by_hover_v1(
            firefly_df, arm_df, ulg_dict)
        # [firefly_df, arm_df, ulg_dict] = FUParser.filter_by_hover_v2(
        #     firefly_df, arm_df, ulg_dict)
        # [firefly_df, arm_df, ulg_dict] = FUParser.filter_by_hover_v3(
        #     firefly_df, arm_df, ulg_dict)

        # power vs (time, eta_rpm, delta_rpm) using filtered hover data
        self.fup_power.ver1(firefly_df, arm_df, ulg_dict, file_tag, ['img16'])
        self.fup_power.ver2(firefly_df, arm_df, ulg_dict, file_tag, ['img17'])
        self.fup_power.ver3(firefly_df, arm_df, ulg_dict, file_tag, ['img18'])
        self.fup_power.ver4(firefly_df, arm_df, ulg_dict, file_tag, ['img19'])
        self.fup_power.jack(firefly_df, arm_df, ulg_dict, file_tag, ['img20'])

        # print(f'firefly_df.keys() {firefly_df.keys()}')
        # print(f'arm_df.keys() {arm_df.keys()}')
        # for k in ulg_dict.keys():
        #     ulg_df = ulg_dict[k]
        #     print(f'ulg_dict[{k}]')
        #     print(f'ulg_df.keys() {ulg_df.keys()}')

    def save_current_plot(self, file_tag, tag_arr, sep, ext):
        file_name = file_tag
        for tag in tag_arr:
            file_name = file_name + sep + str(tag)
        file_path = self.plotdir + f'/' + file_name + ext

        # plt.show()
        print(f'Saving file {file_path} ..')
        plt.savefig(file_path)
        plt.close(plt.gcf())
        # return file_path

    @staticmethod
    def set_axes_limits(axs, xlims, ylims):
        i = 0
        for ax in axs:
            xmin_xmax = xlims[i]
            if len(xmin_xmax) == 2:
                xmin = xmin_xmax[0]
                xmax = xmin_xmax[1]
                if (xmin is not None) and (xmax is not None):
                    ax.set_xlim([xmin, xmax])
            ymin_ymax = ylims[i]
            if len(ymin_ymax) == 2:
                ymin = ymin_ymax[0]
                ymax = ymin_ymax[1]
                if (ymin is not None) and (ymax is not None):
                    ax.set_ylim([ymin, ymax])
            i = i + 1


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Parse, process and plot .kdecan files')
    parser.add_argument('--bdir', action='store', required=True,
                        help='Base directory of [logs, tmp, plots] folders')
    parser.add_argument('--ulg', action='store', required=True,
                        help='Specific log file number to process')
    parser.add_argument('--firefly', action='store', required=True,
                        help='Specific log file number to process')

    args = parser.parse_args()

    abs_bdir = os.path.abspath(args.bdir)
    if (args.firefly is not None) and (args.ulg is not None):
        abs_firefly_file = find_file_in_folder(
            f'{abs_bdir}/logs', '.firefly', args.firefly)
        abs_ulg_file = find_file_in_folder(
            f'{abs_bdir}/logs', '.ulg', args.ulg)
        print(f'firefly file: {abs_firefly_file}')
        print(f'ulg file: {abs_ulg_file}')

        fu_plot = FUPlot(abs_bdir)
        abs_fu_tag = f'firefly_log_{args.firefly}_ulg_log_{args.ulg}'
        fu_plot.process_file(abs_firefly_file, abs_ulg_file, abs_fu_tag)
        exit(0)

    print('Error parsing user arguments')
    print(f'firefly file: {args.firefly}')
    print(f'kdecan file: {args.kdecan}')
    print(f'ulg file: {args.ulg}')
    exit(0)

    # python firefly_plot_fu.py
    # --bdir /home/tzo4/Dropbox/tomas/pennState_avia/firefly_logBook/
    # 2021-12-21_hangar/ --firefly 6 --ulg 135

    # python firefly_plot_fu.py
    # --bdir /home/tzo4/Dropbox/tomas/pennState_avia/firefly_logBook/
    # 2021-12-10_hangar/ --firefly 54 --ulg 109

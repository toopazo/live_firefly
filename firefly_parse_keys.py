
class FireflyDfKeysMi:
    def __init__(self, mi):
        mi = int(mi)
        if mi not in [1, 2, 3, 4, 5, 6, 7, 8]:
            raise RuntimeError
        self.mi = str(mi)

        # ['ars_mills', 'ars_secs',
        #      'ars_rpm1', 'ars_rpm2', 'ars_rpm3', 'ars_rpm4',
        #      'ars_rpm5', 'ars_rpm6', 'ars_rpm7', 'ars_rpm8',
        #      'ars_cur1', 'ars_cur2', 'ars_cur3', 'ars_cur4',
        #      'ars_cur5', 'ars_cur6', 'ars_cur7', 'ars_cur8',
        #      'ars_rpm_13', 'ars_rpm_18', 'ars_rpm_14', 'ars_rpm_17',
        #      'ars_cur_13', 'ars_cur_18', 'ars_cur_14', 'ars_cur_17',
        #      'esc11_escid', 'esc11_voltage', 'esc11_current', 'esc11_rpm',
        #      'esc11_inthrottle', 'esc11_outthrottle',
        #      'esc12_escid', 'esc12_voltage', 'esc12_current', 'esc12_rpm',
        #      'esc12_inthrottle', 'esc12_outthrottle',
        #      'esc13_escid', 'esc13_voltage', 'esc13_current', 'esc13_rpm',
        #      'esc13_inthrottle', 'esc13_outthrottle',
        #      'esc14_escid', 'esc14_voltage', 'esc14_current',
        #  'esc14_rpm', 'esc14_inthrottle', 'esc14_outthrottle', 'esc15_escid',
        #  'esc15_voltage', 'esc15_current', 'esc15_rpm', 'esc15_inthrottle',
        #  'esc15_outthrottle', 'esc16_escid', 'esc16_voltage', 'esc16_current',
        #  'esc16_rpm', 'esc16_inthrottle', 'esc16_outthrottle', 'esc17_escid',
        #  'esc17_voltage', 'esc17_current', 'esc17_rpm', 'esc17_inthrottle',
        #  'esc17_outthrottle', 'esc18_escid', 'esc18_voltage', 'esc18_current',
        #  'esc18_rpm', 'esc18_inthrottle', 'esc18_outthrottle',

        self.thr = f'esc1{self.mi}_inthrottle'
        self.rpm = f'esc1{self.mi}_rpm'
        self.cur = f'esc1{self.mi}_current'
        self.vol = f'esc1{self.mi}_voltage'
        # ars_cur_14
        self.cur_ars = f'ars_cur_1{self.mi}'

        #      'fcost_fcost1', 'fcost_fcost2', 'fcost_fcost3', 'fcost_fcost4',
        #      'fcost_fcost5', 'fcost_fcost6', 'fcost_fcost7', 'fcost_fcost8',
        #      'optim_nsh_cmd', 'optim_avg_cost_m38',
        #      'optim_avg_cost_m47', 'optim_avg_cost_tot',
        #      'optim_avg_cost_tot_prev']

        self.fcost = f'fcost_fcost{mi}'
        # self.nsh_cmd = 'optim_nsh_cmd'
        # self.fcost_avg_m38 = 'optim_avg_cost_m38'
        # self.fcost_avg_m47 = 'optim_avg_cost_m47'
        # self.fcost_avg_tot = 'optim_avg_cost_tot'
        # self.fcost_avg_tot_prev = 'optim_avg_cost_tot_prev'


class FireflyDfKeys:
    m1 = FireflyDfKeysMi(1)
    m2 = FireflyDfKeysMi(2)
    m3 = FireflyDfKeysMi(3)
    m4 = FireflyDfKeysMi(4)
    m5 = FireflyDfKeysMi(5)
    m6 = FireflyDfKeysMi(6)
    m7 = FireflyDfKeysMi(7)
    m8 = FireflyDfKeysMi(8)

    nsh_cmd = 'optim_nsh_cmd'
    fcost_avg_m38 = 'optim_avg_cost_m38'
    fcost_avg_m47 = 'optim_avg_cost_m47'
    fcost_avg_tot = 'optim_avg_cost_tot'
    fcost_avg_tot_prev = 'optim_avg_cost_tot_prev'


class ArmDfKeysMi:
    def __init__(self, mi):
        mi = int(mi)
        if mi not in [1, 2, 3, 4, 5, 6, 7, 8, 16, 25, 38, 47]:
            raise RuntimeError
        self.mi = str(mi)

        # 'm16_eta_thr', 'm25_eta_thr', 'm38_eta_thr', 'm47_eta_thr',
        # 'm16_eta_rpm', 'm25_eta_rpm', 'm38_eta_rpm', 'm47_eta_rpm',
        # 'm16_eta_cur', 'm25_eta_cur', 'm38_eta_cur', 'm47_eta_cur',
        self.eta_thr = f'm{mi}_eta_thr'
        self.eta_rpm = f'm{mi}_eta_rpm'
        self.eta_cur = f'm{mi}_eta_cur'

        # 'm16_delta_thr', 'm25_delta_thr', 'm38_delta_thr', 'm47_delta_thr',
        # 'm16_delta_rpm', 'm25_delta_rpm', 'm38_delta_rpm', 'm47_delta_rpm',
        # 'm16_delta_cur', 'm25_delta_cur', 'm38_delta_cur', 'm47_delta_cur',
        self.delta_thr = f'm{mi}_delta_thr'
        self.delta_rpm = f'm{mi}_delta_rpm'
        self.delta_cur = f'm{mi}_delta_cur'

        # 'm1_rate_thr', 'm2_rate_thr', 'm3_rate_thr', 'm4_rate_thr',
        # 'm5_rate_thr', 'm6_rate_thr', 'm7_rate_thr', 'm8_rate_thr',
        # 'm1_rate_rpm', 'm2_rate_rpm', 'm3_rate_rpm', 'm4_rate_rpm',
        # 'm5_rate_rpm', 'm6_rate_rpm', 'm7_rate_rpm', 'm8_rate_rpm',
        # 'm1_rate_cur', 'm2_rate_cur', 'm3_rate_cur', 'm4_rate_cur',
        # 'm5_rate_cur', 'm6_rate_cur', 'm7_rate_cur', 'm8_rate_cur',
        # 'm16_rate_thr', 'm25_rate_thr', 'm38_rate_thr', 'm47_rate_thr',
        # 'm16_rate_rpm', 'm25_rate_rpm', 'm38_rate_rpm', 'm47_rate_rpm',
        # 'm16_rate_cur', 'm25_rate_cur', 'm38_rate_cur', 'm47_rate_cur',
        self.rate_thr = f'm{mi}_rate_thr'
        self.rate_rpm = f'm{mi}_rate_rpm'
        self.rate_cur = f'm{mi}_rate_cur'

        # 'm1_pow_esc', 'm2_pow_esc', 'm3_pow_esc', 'm4_pow_esc', 'm5_pow_esc',
        # 'm6_pow_esc', 'm7_pow_esc', 'm8_pow_esc', 'm16_pow_esc', 'm25_pow_esc'
        # 'm38_pow_esc', 'm47_pow_esc', 'm1_pow_ars', 'm2_pow_ars', 'm3_pow_ars'
        # 'm4_pow_ars', 'm5_pow_ars', 'm6_pow_ars', 'm7_pow_ars', 'm8_pow_ars',
        # 'm16_pow_ars', 'm25_pow_ars', 'm38_pow_ars', 'm47_pow_ars'
        self.pow_esc = f'm{mi}_pow_esc'
        self.pow_ars = f'm{mi}_pow_ars'


class ArmDfKeys:
    m1 = ArmDfKeysMi(1)
    m2 = ArmDfKeysMi(2)
    m3 = ArmDfKeysMi(3)
    m4 = ArmDfKeysMi(4)
    m5 = ArmDfKeysMi(5)
    m6 = ArmDfKeysMi(6)
    m7 = ArmDfKeysMi(7)
    m8 = ArmDfKeysMi(8)

    m16 = ArmDfKeysMi(16)
    m25 = ArmDfKeysMi(25)
    m38 = ArmDfKeysMi(38)
    m47 = ArmDfKeysMi(47)


class UlgDictKeys:
    ulg_pv_df = 'ulg_pv_df'
    ulg_accel_df = 'ulg_accel_df'
    ulg_att_df = 'ulg_att_df'
    ulg_attsp_df = 'ulg_attsp_df'
    ulg_angvel_df = 'ulg_angvel_df'
    ulg_angvelsp_df = 'ulg_angvelsp_df'
    ulg_angacc_df = 'ulg_angacc_df'
    ulg_in_df = 'ulg_in_df'
    ulg_out_df = 'ulg_out_df'


class UlgPvDfKeys:
    # ulg_dict[ulg_pv_df]
    # ['x', 'y', 'z', 'vx', 'vy', 'vz', 'vnorm', 'pnorm']
    x = 'x'
    y = 'y'
    z = 'z'
    vx = 'vx'
    vy = 'vy'
    vz = 'vz'
    vel_norm = 'vnorm'
    pos_norm = 'pnorm'


class UlgAccelDf:
    # ulg_dict[ulg_accel_df]
    # ['ax', 'ay', 'az', 'acc norm']
    ax = 'ax'
    ay = 'ay'
    az = 'az'
    acc_norm = 'acc norm'


class UlgAttDf:
    # ulg_dict[ulg_att_df]
    # ['roll', 'pitch', 'yaw', 'angle norm']
    roll = 'roll'
    pitch = 'pitch'
    yaw = 'yaw'
    angle_norm = 'angle norm'


class UlgAttspDf:
    # ulg_dict[ulg_attsp_df]
    # ['roll sp', 'pitch sp', 'yaw sp', 'angle norm sp']
    roll = 'roll sp'
    pitch = 'pitch sp'
    yaw = 'yaw sp'
    angle_norm = 'angle norm sp'


class UlgAngvelDf:
    # ulg_dict[ulg_angvel_df]
    # ['roll rate', 'pitch rate', 'yaw rate', 'pqr norm']
    roll_rate = 'roll rate'
    pitch_rate = 'pitch rate'
    yaw_rate = 'yaw rate'
    pqr_norm = 'pqr norm'


class UlgAngvelspDf:
    # ulg_dict[ulg_angvelsp_df]
    # ['roll rate sp', 'pitch rate sp', 'yaw rate sp', 'pqr norm sp']
    roll_rate_sp = 'roll rate sp'
    pitch_rate_sp = 'pitch rate sp'
    yaw_rate_sp = 'yaw rate sp'
    pqr_norm_sp = 'pqr norm sp'


class UlgAngaccDf:
    # ulg_dict[ulg_angacc_df]
    p_rate = 'p rate'
    q_rate = 'q rate'
    r_rate = 'r rate'
    angacc_norm = 'angacc norm'


class UlgInDfKeys:
    # ulg_dict[ulg_in_df]
    # ['p rate cmd', 'q rate cmd', 'r rate cmd', 'az cmd']
    p_rate_cmd = 'p rate cmd'
    q_rate_cmd = 'q rate cmd'
    r_rate_cmd = 'r rate cmd'
    az_cmd = 'az cmd'


class UlgOutDfKeys:
    # ulg_dict[ulg_out_df]
    output_0 = 'output[0]'
    output_1 = 'output[1]'
    output_2 = 'output[2]'
    output_3 = 'output[3]'
    output_4 = 'output[4]'
    output_5 = 'output[5]'
    output_6 = 'output[6]'
    output_7 = 'output[7]'

# From file mixer_module.cpp located in
#
#   /home/tzo4/Dropbox/tomas/pennState_avia/software/px4_sourcecode/
#   PX4-Autopilot/src/lib/mixer_module/
#
# a custom CA was implemented, where B+ = B8pinvn and is given by
#
# float B8pinvn[8][8] = {
#     {-1.4142, +1.4142, +2.0000, +2.0000, +0.4981, +0.0019, -0.0019, +0.0019},
#     {+1.4142, +1.4142, -2.0000, +2.0000, +0.0019, +0.4981, +0.0019, -0.0019},
#     {+1.4142, -1.4142, +2.0000, +2.0000, -0.0019, +0.0019, +0.4981, +0.0019},
#     {-1.4142, -1.4142, -2.0000, +2.0000, +0.0019, -0.0019, +0.0019, +0.4981},
#     {+1.4142, +1.4142, +2.0000, +2.0000, -0.0019, -0.4981, -0.0019, +0.0019},
#     {-1.4142, +1.4142, -2.0000, +2.0000, -0.4981, -0.0019, +0.0019, -0.0019},
#     {-1.4142, -1.4142, +2.0000, +2.0000, -0.0019, +0.0019, -0.0019, -0.4981},
#     {+1.4142, -1.4142, -2.0000, +2.0000, +0.0019, -0.0019, -0.4981, -0.0019}
#     };
#
# The matrix multiplication is carried out by the code
#
# for (unsigned i = 0; i < 8; i++) {
#     unsigned rowi = i;
#     outputs[rowi] =
#         B8pinvn[rowi][0] * controls.control[0] +
#         B8pinvn[rowi][1] * controls.control[1] +
#         B8pinvn[rowi][2] * controls.control[2] +
#         B8pinvn[rowi][3] * controls.control[3] +
#         B8pinvn[rowi][4] * delta.delta[0] +
#         B8pinvn[rowi][5] * delta.delta[1] +
#         B8pinvn[rowi][6] * delta.delta[2] +
#         B8pinvn[rowi][7] * delta.delta[3] +
#         - 1;
#
# This matrix was obtained from
#    /home/tzo4/Dropbox/tomas/pennState_avia/software/px4_ctrlalloc/
#    px4_mixer/ctrlalloc_octocoax_px4.m
#
# The contents below are the Python version of that file
#

import numpy as np
# import scipy
import scipy.linalg
import pprint


# clear all
# clc
# close all
# format compact
# format short

def print_matrix(matrix, dec):
    matrix = np.around(matrix, decimals=dec)
    s = [[str(f'%+f' % e) for e in row] for row in matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))


def plot_arms(thr0, rpm0):
    # print('                   ||| ')
    # print('%.04f (%.04f) ------------- %.04f (%.04f) ' % (thr0[1], thr0[0]))
    # print('%.04f (%.04f) ------------- %.04f (%.04f) ' % (thr0[4], thr0[5]))
    # print('                   ||| ')
    # print('                   ||| ')
    # print('                   ||| ')
    # print('                   ||| ')
    # print('%.04f (%.04f) ------------- %.04f (%.04f) ' % (thr0[2], thr0[3]))
    # print('%.04f (%.04f) ------------- %.04f (%.04f) ' % (thr0[7], thr0[6]))
    # print('Inside plot_arms ..')

    m2_m1 = thr0[2-1], rpm0[2-1], thr0[1-1], rpm0[1-1]
    m5_m6 = thr0[5-1], rpm0[5-1], thr0[6-1], rpm0[6-1]
    m3_m4 = thr0[3-1], rpm0[3-1], thr0[4-1], rpm0[4-1]
    m8_m7 = thr0[8-1], rpm0[8-1], thr0[7-1], rpm0[7-1]
    print('                        ||| ')
    print('%+.04f (%i rpm) ------------- %+.04f (%i rpm) ' % m2_m1)
    print('%+.04f (%i rpm) ------------- %+.04f (%i rpm) ' % m5_m6)
    print('                        ||| ')
    print('                        ||| ')
    print('                        ||| ')
    print('                        ||| ')
    print('%+.04f (%i rpm) ------------- %+.04f (%i rpm) ' % m3_m4)
    print('%+.04f (%i rpm) ------------- %+.04f (%i rpm) ' % m8_m7)


def ctrl_alloc_using_pinv(a8_px4, a8pinv_px4, b_cmd, dec):
    print('')
    print('a8pinv_px4')
    print_matrix(a8pinv_px4, dec=dec)
    print('')
    print('b_cmd')
    print_matrix(b_cmd, dec=dec)

    ca_x0 = np.matmul(a8pinv_px4, b_cmd)  # A8pinv_px4 * b
    thr0 = ca_x0 - 1
    rpm0 = 1650 * thr0 + 2350
    delta_rpm0 = [rpm0[1-1] - rpm0[6-1], rpm0[2-1] - rpm0[5-1],
                  rpm0[3-1] - rpm0[8-1], rpm0[4-1] - rpm0[7-1]]
    # x0 = linsolve(A8pinv_px4, b)
    # x0 = lsqnonneg(A8pinv_px4, b)
    ca_solution_err = np.matmul(a8_px4, ca_x0) - b_cmd
    ca_norm_sol_err = np.linalg.norm(ca_solution_err)

    print('')
    print('x0')
    print_matrix(ca_x0, dec=dec)
    print('')
    print('a8_px4 * x0 - b      = solution error = ')
    print_matrix(ca_solution_err, dec=dec)
    print('')
    print('| a8_px4 * x0 - b |  = norm sol error =')
    print(ca_norm_sol_err)

    print('')
    print('thr0')
    print_matrix(thr0, dec=dec)
    # print('')
    # print('rpm0')
    # print_matrix(rpm0, dec=dec)
    print('')
    print('delta_rpm0')
    print_matrix(delta_rpm0, dec=dec)

    print('')
    plot_arms(thr0, rpm0)
    
    return ca_norm_sol_err
    
    
def ctrl_alloc_using_pinv_noprint(a8_px4, a8pinv_px4, b_cmd, dec):
    ca_x0 = np.matmul(a8pinv_px4, b_cmd)  # A8pinv_px4 * b
    thr0 = ca_x0 - 1
    rpm0 = 1650 * thr0 + 2350
    delta_rpm0 = [rpm0[1-1] - rpm0[6-1], rpm0[2-1] - rpm0[5-1],
                  rpm0[3-1] - rpm0[8-1], rpm0[4-1] - rpm0[7-1]]
    # x0 = linsolve(A8pinv_px4, b)
    # x0 = lsqnonneg(A8pinv_px4, b)
    ca_solution_err = np.matmul(a8_px4, ca_x0) - b_cmd
    ca_norm_sol_err = np.linalg.norm(ca_solution_err)
    
    return ca_norm_sol_err    
    
    
def fzero(x, a8_px4, a8pinv_px4):
    udec = 8
    Pdot = 0.10
    Qdot = 0.20
    Rdot = x
    Wdot = 0.50
    d1 = -0.15 * 1
    d2 = -0.25 * 1
    d3 = -0.35 * 1
    d4 = -0.45 * 1
    b_des = np.array([[Pdot, Qdot, Rdot, Wdot, d1, d2, d3, d4]])
    b_des = b_des.transpose()
    ca_norm_sol_err = ctrl_alloc_using_pinv_noprint(
        a8_px4=A8_px4, a8pinv_px4=A8pinv_px4, b_cmd=b_des, dec=udec)
    return ca_norm_sol_err


if __name__ == '__main__':
    udec = 8

    pprint.pprint('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    pprint.pprint('Control allocation as defined in px4')
    pprint.pprint('B8_px4     = 4x8 = From omega_cmd    to PQRW_cmd')
    pprint.pprint('B8pinv_px4 = 8x4 = From PQRW_cmd     to omega_cmd')
    pprint.pprint('A8_px4     = 8x8 = From omega_cmd    to PQRWdddd_cmd')
    pprint.pprint('A8pinv_px4 = 8x8 = From PQRWdddd_cmd to omega_cmd')

    # Obtained from ulg_plot_mixer.py and Houselfy log 148
    B8pinv_px4 = [
        [-1.4142, 1.4142, 2.0000, 2.0000],
        [1.4142, 1.4142, -2.0000, 2.0000],
        [1.4142, -1.4142, 2.0000, 2.0000],
        [-1.4142, -1.4142, -2.0000, 2.0000],
        [1.4142, 1.4142, 2.0000, 2.0000],
        [-1.4142, 1.4142, -2.0000, 2.0000],
        [-1.4142, -1.4142, 2.0000, 2.0000],
        [1.4142, -1.4142, -2.0000, 2.0000],
    ]
    pprint.pprint('B8pinv_px4')
    print_matrix(B8pinv_px4, dec=udec)

    # From /build/px4_fmu-v5_default/
    # src/lib/mixer/mixer_multirotor_normalized.generated.h
    # B8pinv_px4 = [
    # 	[ -0.707107,  0.707107,  1.000000,  1.000000 ],
    # 	[  0.707107,  0.707107, -1.000000,  1.000000 ],
    # 	[  0.707107, -0.707107,  1.000000,  1.000000 ],
    # 	[ -0.707107, -0.707107, -1.000000,  1.000000 ],
    # 	[  0.707107,  0.707107,  1.000000,  1.000000 ],
    # 	[ -0.707107,  0.707107, -1.000000,  1.000000 ],
    # 	[ -0.707107, -0.707107,  1.000000,  1.000000 ],
    # 	[  0.707107, -0.707107, -1.000000,  1.000000 ],
    # ]

    B8_px4 = np.linalg.pinv(B8pinv_px4)
    pprint.pprint('B8_px4')
    print_matrix(B8_px4, dec=udec)
    pprint.pprint(f'B8_px4.shape {B8_px4.shape}')
# -0.0884    0.0884    0.0884   -0.0884    0.0884   -0.0884   -0.0884    0.0884
#  0.0884    0.0884   -0.0884   -0.0884    0.0884    0.0884   -0.0884   -0.0884
#  0.0625   -0.0625    0.0625   -0.0625    0.0625   -0.0625    0.0625   -0.0884
#  0.0625    0.0625    0.0625    0.0625    0.0625    0.0625    0.0625    0.0884

    sf = 1  # 0.0625;  # Scale factor
    # A8_px4 = [
    #     B8_px4,
    #     sf * [+1, +0, +0, +0, +0, -1, +0, +0],
    #     sf * [+0, +1, +0, +0, -1, +0, +0, +0],
    #     sf * [+0, +0, +1, +0, +0, +0, +0, -1],
    #     sf * [+0, +0, +0, +1, +0, +0, -1, +0],
    # ]
    A8_px4 = np.append(B8_px4,  [sf * [+1, +0, +0, +0, +0, -1, +0, +0]], axis=0)
    A8_px4 = np.append(A8_px4,  [sf * [+0, +1, +0, +0, -1, +0, +0, +0]], axis=0)
    A8_px4 = np.append(A8_px4,  [sf * [+0, +0, +1, +0, +0, +0, +0, -1]], axis=0)
    A8_px4 = np.append(A8_px4,  [sf * [+0, +0, +0, +1, +0, +0, -1, +0]], axis=0)
    pprint.pprint('A8_px4')
    print_matrix(A8_px4, dec=udec)
# -0.0884    0.0884    0.0884   -0.0884    0.0884   -0.0884   -0.0884    0.0884
# 0.0884    0.0884   -0.0884   -0.0884    0.0884    0.0884   -0.0884   -0.0884
# 0.0625   -0.0625    0.0625   -0.0625    0.0625   -0.0625    0.0625   -0.0625
# 0.0625    0.0625    0.0625    0.0625    0.0625    0.0625    0.0625    0.0625
# 1.0000         0         0         0         0   -1.0000         0         0
#     0    1.0000         0         0   -1.0000         0         0         0
#     0         0    1.0000         0         0         0         0   -1.0000
#     0         0         0    1.0000         0         0   -1.0000         0

    A8pinv_px4 = np.linalg.pinv(A8_px4)
    pprint.pprint('A8pinv_px4')
    print_matrix(A8pinv_px4, dec=udec)
# -1.4142    1.4142    0.0308    2.0000    0.4981    0.0019   -0.0019    0.0019
# 1.4142    1.4142   -0.0308    2.0000    0.0019    0.4981    0.0019   -0.0019
# 1.4142   -1.4142    0.0308    2.0000   -0.0019    0.0019    0.4981    0.0019
# -1.4142   -1.4142   -0.0308    2.0000    0.0019   -0.0019    0.0019    0.4981
# 1.4142    1.4142    0.0308    2.0000   -0.0019   -0.4981   -0.0019    0.0019
# -1.4142    1.4142   -0.0308    2.0000   -0.4981   -0.0019    0.0019   -0.0019
# -1.4142   -1.4142    0.0308    2.0000   -0.0019    0.0019   -0.0019   -0.4981
# 1.4142   -1.4142   -0.0308    2.0000    0.0019   -0.0019   -0.4981   -0.0019

# A8pinv_px4(:,3) = A8pinv_px4(:,3).*64.999;  # makes 3rd column (yaw) == 2.0
# A8pinv_px4
# -1.4142    1.4142    2.0000    2.0000    0.4981    0.0019   -0.0019    0.0019
# 1.4142    1.4142   -2.0000    2.0000    0.0019    0.4981    0.0019   -0.0019
# 1.4142   -1.4142    2.0000    2.0000   -0.0019    0.0019    0.4981    0.0019
# -1.4142   -1.4142   -2.0000    2.0000    0.0019   -0.0019    0.0019    0.4981
# 1.4142    1.4142    2.0000    2.0000   -0.0019   -0.4981   -0.0019    0.0019
# -1.4142    1.4142   -2.0000    2.0000   -0.4981   -0.0019    0.0019   -0.0019
# -1.4142   -1.4142    2.0000    2.0000   -0.0019    0.0019   -0.0019   -0.4981
# 1.4142   -1.4142   -2.0000    2.0000    0.0019   -0.0019   -0.4981   -0.0019

    check_geninv = np.linalg.norm(
        np.matmul(
            np.matmul(A8_px4, A8pinv_px4), A8_px4
        )
        - A8_px4
    )
    pprint.pprint('check_geninv')
    pprint.pprint(check_geninv)
    if check_geninv > 10**-5:
        pprint.pprint('check_geninv > 10**-5')
        raise RuntimeError

    pprint.pprint('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    pprint.pprint('Actuator solutions')

    Pdot = 0.10
    Qdot = 0.20
    Rdot = 0.30
    Wdot = 0.40
    d1 = -0.15 * 1
    d2 = -0.25 * 1
    d3 = -0.35 * 1
    d4 = -0.45 * 1
    b_des = np.array([[Pdot, Qdot, Rdot, Wdot, d1, d2, d3, d4]])
    b_des = b_des.transpose()
    pprint.pprint('b')
    print_matrix(b_des, dec=udec)

    sizeA = np.shape(A8_px4)
    rankA = np.linalg.matrix_rank(A8_px4)
    # rankAb = np.linalg.matrix_rank([A8_px4, b])
    # pprint.pprint('A8_px4.shape {A8_px4.shape}')
    # pprint.pprint('b.shape {b.shape}')
    A8_px4_b = np.append(A8_px4, b_des, axis=1)
    # print_matrix(A8_px4_b, dec=num_dec)
    rankAb = np.linalg.matrix_rank(A8_px4_b)
    pprint.pprint(f'sizeA {sizeA}')
    pprint.pprint(f'rankA {rankA}')
    pprint.pprint(f'rankAb {rankAb}')
    if rankAb > rankA:
        pprint.pprint('Inconsistent system, rankAb > rankA')
        # raise RuntimeError
    else:
        pprint.pprint('Sys. is consistent but rank deficient => inf solutions')

    # k_dof = length(omega8) - rankAb
    detA = np.linalg.det(A8_px4)
    nullA = scipy.linalg.null_space(A8_px4)
    pprint.pprint(f'detA {detA}')
    pprint.pprint('nullA')
    print_matrix(nullA / np.abs(nullA), dec=udec)

    pprint.pprint('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    pprint.pprint('Actuator solution using Pseudoinverse')
    A8pinv_px4 = np.matmul(np.transpose(A8_px4), np.linalg.inv(np.matmul(A8_px4, np.transpose(A8_px4))))    
    b_des[2] = (d1 - d2 + d3 - d4) * 0.062500
    ctrl_alloc_using_pinv(
        a8_px4=A8_px4, a8pinv_px4=A8pinv_px4, b_cmd=b_des, dec=udec)

#    pprint.pprint('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
#    pprint.pprint('Actuator solution using modified Pseudoinverse')
#    # make 3rd column (yaw) == 2.0
#    # A8pinv_px4[:, 2] = A8pinv_px4[:, 2] * 64.9999
#    A8pinv_px4[:, 2] = [+2, -2, +2, -2, +2, -2, +2, -2]
#    A8pinv_px4 = np.matmul(np.transpose(A8_px4), np.linalg.inv(np.matmul(A8_px4, np.transpose(A8_px4))))    
#    ctrl_alloc_using_pinv(
#        a8_px4=A8_px4, a8pinv_px4=A8pinv_px4, b_cmd=b_des, dec=udec)
        
#    pprint.pprint('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
#    pprint.pprint('Test fzero')        
#    # err = fzero(x=1, a8_px4=A8_px4, a8pinv_px4=A8pinv_px4)        
#    # print(err)
#    # err = fzero(x=2, a8_px4=A8_px4, a8pinv_px4=A8pinv_px4)        
#    # print(err)
#    from scipy.optimize import fsolve
#    root = fsolve(fzero, 0.1, args=(A8_px4, A8pinv_px4))
#    print(root)

#    pprint.pprint('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
#    pprint.pprint('Actuator solution using default Pseudoinverse')
#    ctrl_alloc_using_pinv(
#        a8_px4=B8_px4, a8pinv_px4=B8pinv_px4, b_cmd=b_des[:4], dec=udec)

    #    disp('Actuator solution using Pseudoinverse + NullSpace')
    #    # x0 = x0 + [zeros(8, 1); 1; 1; 1; 1]
    #    alpha = -0.1; # any scalar will do, because we are adding a null-vector
    #    nullA = null(A8pinv_px4)
    #    x0 = x0 + nullA*alpha
    #    err_Axb = A8_px4*x0 - b;
    #    norm_err_Axb = norm(err_Axb)
    
    
    

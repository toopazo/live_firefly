import matplotlib.pyplot as plt
import numpy as np


class OptimNoise:
    a = 0.0001 / 25
    b = 0.1 / 25
    c = 25 / 25
    awgn = (c / 300)
    awgn_flag = True

    def __init__(self):
        [i_arr, xi_arr, fc_arr] = OptimNoise.optimize()
        OptimNoise.plot_results(i_arr, xi_arr, fc_arr)

    @staticmethod
    def plot_results(i_arr, xi_arr, fc_arr):
        w_inches = 8
        h_inches = 9
        figsize = (w_inches, h_inches)
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=figsize)

        nsteps = len(i_arr)
        xi_mean = np.mean(xi_arr[10:])
        xi_std = np.std(xi_arr[10:])
        print(f'xi stats after 10 steps: {xi_mean} {xi_std}')
        fc_mean = np.mean(fc_arr[10:])
        fc_std = np.std(fc_arr[10:])
        print(f'fc stats after 10 steps: {fc_mean} {fc_std}')

        ax1.grid(True)
        ax1.set_ylabel(r'Input evolution')
        ax1.plot(i_arr, xi_arr,
                 label='input', marker='.', color='blue', markersize=9)
        ax1.set_xlabel("Steps")
        # ax1.axes.xaxis.set_ticklabels([])
        ax1.set_xlim([0, nsteps])
        ax1.set_ylim([-1000, 0])
        ax1.set_yticks([-1000, -500, 0])
        # ax1.set_xticks(np.linspace(0, nsteps, 10))
        ax1.legend(loc='upper right')

        ax2 = ax1.twinx()
        ax2.grid(True)
        ax2.set_ylabel(r'Cost evolution')
        ax2.plot(i_arr, fc_arr,
                 label='cost', marker='.', color='red', markersize=9)
        ax2.set_xlabel("Steps")
        ax2.set_ylim([-0.5, 1.5])
        ax2.set_yticks([-0.5, 0, 0.5, 1.0, 1.5])
        ax2.legend(loc='lower right')

        xarr = np.linspace(-1000, 0, 100)
        farr = [OptimNoise.fcost(x) for x in xarr]
        lbl = f'num steps {nsteps}'

        ax3.grid(True)
        ax3.set_ylabel(r'Cost function')
        ax3.plot(xarr, farr, label='')
        ax3.scatter(xi_arr, fc_arr, label=lbl, marker='.', color='red', s=80)
        ax3.set_xlabel("Input variable")
        ax3.set_xlim([-1000, 0])
        ax3.set_ylim([0, 1])

        a = OptimNoise.a
        b = OptimNoise.b
        c = OptimNoise.c
        awgn = OptimNoise.awgn
        awgn_flag = OptimNoise.awgn_flag
        if awgn_flag:
            textstr = f'f(x) = {a} x^2 + {b} x + {c} \n' \
                      f'AWGN of magnitude {round(awgn, 4)}'
        else:
            textstr = f'f(x) = {a} x^2 + {b} x + {c} \n' \
                      f'No noise'
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # place a text box in upper left in axes coords
        # axt.text(1110, 0.066, textstr, fontsize=10,
        ax3.text(-750, 0.8, textstr, fontsize=15,
                 verticalalignment='top', bbox=props)

        file_path = '/home/tzo4/Dropbox/tomas/pennState_avia/firefly_logBook/' \
                    '2022-01-20_database/optim_and_noise.png'
        print(f'Saving file {file_path} ..')
        plt.savefig(file_path, bbox_inches='tight')
        plt.close(plt.gcf())

    @staticmethod
    def fcost(x):
        a = OptimNoise.a
        b = OptimNoise.b
        c = OptimNoise.c
        x2 = x ** 2
        y = a * x2 + b * x + c

        awgn_flag = OptimNoise.awgn_flag
        if awgn_flag:
            awgn = OptimNoise.awgn
            noise = np.random.normal(0, 1, 1)
            noise = noise[0] * awgn
            y = y + noise

        return y

    @staticmethod
    def fcost_gradient(x):
        dx = 1
        f_x1 = OptimNoise.fcost(x+dx)
        f_x0 = OptimNoise.fcost(x)
        df_dx = (f_x1 - f_x0) / dx
        return df_dx

    @staticmethod
    def optimize():
        x0 = 0
        learn_rate = 20000

        awgn_flag = OptimNoise.awgn_flag
        if awgn_flag:
            nsteps = 61
        else:
            nsteps = 21

        xi = x0
        i_arr = []
        xi_arr = []
        fc_arr = []
        for i in range(0, nsteps):
            # df_dx = OptimNoise.fcost_gradient(xi)
            # fc = OptimNoise.fcost(xi)

            dx = 1
            f_xn = OptimNoise.fcost(xi + dx)
            f_xi = OptimNoise.fcost(xi)
            df_dx = (f_xn - f_xi) / dx
            fc = f_xi

            print(f'iteration {i}, '
                  f'xi {np.around(xi, 2)}, '
                  f'df_dx, {np.around(df_dx, 2)}, '
                  f'fcost {np.around(fc, 2)}')

            i_arr.append(i)
            xi_arr.append(xi)
            fc_arr.append(fc)

            xi = xi - learn_rate * df_dx
            # xi = xi - fc * df_dx

        return [i_arr, xi_arr, fc_arr]


if __name__ == '__main__':
    on = OptimNoise()


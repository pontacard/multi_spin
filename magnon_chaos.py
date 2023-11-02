import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from typeZ import Spin_wave

class Phase_space(Spin_wave):
    def __init__(self,alpha,gamma,B,S0,t,t_eval,Kx,Ky,Kz,beta,pulse,N,J):
        super().__init__(alpha, gamma, B, S0, t, t_eval, Kx,Ky,Kz,beta,pulse,N,J)

    def make_phase_graph(self,i,ax1,ax2):      #二次元のぐらふを作る。iはi番目のスピン、axはどの軸をとるか
        self.S0 = np.reshape(self.S0, (1, -1))
        self.S0 = list(self.S0[0])
        #self.fig, self.ax = plt.subplots(subplot_kw=dict(projection="3d"))
        print(self.S0)

        self.Sol = sc.integrate.solve_ivp(self.func_S, self.t, self.S0, t_eval=self.t_eval)
        self.S = self.Sol.y
        self.spin_log = np.reshape(self.S, (self.N, 3, -1))
        t = self.t_eval
        x = self.spin_log[i][0]
        y = self.spin_log[i][1]
        z = self.spin_log[i][2]

        plt.plot(x,y)
        plt.xlabel("Sx")
        plt.ylabel("Sy")
        plt.show()

        plt.plot(z,x)
        plt.xlabel("Sz")
        plt.ylabel("Sx")
        plt.show()

        plt.plot(z,y)
        plt.xlabel("Sz")
        plt.ylabel("Sy")
        plt.show()

        plt.plot(t, x)
        plt.xlabel("t")
        plt.ylabel("Sx")
        plt.show()

        N = len(y)   #サンプル数
        f_s = 40000  # サンプリングレート f_s[Hz] (任意)
        dt = 1 / f_s  # サンプリング周期 dt[s]

        y_fft = np.fft.fft(y)  # 離散フーリエ変換
        freq = np.fft.fftfreq(N, d=dt)  # 周波数を割り当てる（※後述）
        Amp = abs(y_fft / (N / 2))  # 音の大きさ（振幅の大きさ）
        plt.plot(freq[1:int(N / 2)], Amp[1:int(N / 2)])  # A-f グラフのプロット
        plt.xscale("log")  # 横軸を対数軸にセット

        #plt.show()



if __name__ == '__main__':
    n = 100
    S0 = np.zeros((n, 3))

    for i in range(n):
        S0[i][0] = 1.01
        S0[i][2] = 0.3


    t = [0,100]  # t(時間)が0〜100まで動き、その時のfを求める。
    t_eval = np.linspace(*t, 20000)

    mu_0 = 1.2
    gamma = 2.8
    h_div_2e = [0.329, -15]
    sta_M = [1.4, 0]  # 飽和磁化(T)で入れる
    theta = [-2.2, -1]
    j = [4.5, 12]
    d = [1.48, -9]
    Hsn = h_div_2e[0] * theta[0] * j[0] / (sta_M[0] * d[0])
    Hso = h_div_2e[1] + theta[1] + j[1] - (sta_M[1] + d[1])
    Hs = Hsn * (10 ** Hso) * 1000 * (mu_0 / 1200000) * mu_0  # 最後の1000はmTにするため

    pulse = []
    for i in range(1):
        tempul = [3 + i, 3 + i + 2, [0, -Hs, 0]]
        pulse.append(tempul)


    spin = Phase_space(0.0001, gamma, [0, 0, mu_0 * 12], S0, t, t_eval, mu_0 * 4,0,- mu_0 * 41.6, 0, pulse, n, 1)
    spin.make_phase_graph(50,1,2)
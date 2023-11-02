import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import ArtistAnimation
from matplotlib.animation import FuncAnimation

class Spin_wave():
    def __init__(self,alpha,gamma,B,S0,t,t_eval,Kx,Ky,Kz,beta,pulse,N,J):
        self.alpha = alpha  # 緩和項をどれくらい大きく入れるか
        self.gamma = gamma  # LLg方程式のγ
        self.B = B  # 外部磁場(tで時間変化させてもいいよ)
        self.S0 = S0  # 初期のスピンの向き
        self.t = t  # どれくらいの時間でやるか
        self.t_eval = t_eval #ステップ数
        self.Kx = Kx #異方性磁気定数
        self.Ky = Ky
        self.Kz = Kz
        self.beta = beta
        self.pulse = pulse #[[開始時刻, 終了時刻, パルスの強さ], [開始時刻, 終了時刻, パルスの強さ],,,,]のようなパルス電流の情報
        self.J = J #交換相互作用定数
        self.N = N # スピンの数
        self.S = []
        self.ax = 0  # pltのオブジェクト
        self.fig = 0  # pltのオブジェクト
        self.quiveraa = 0
        self.quiver_dic = dict()
        self.spin_log = []

    def func_S(self,t,S):
        dSdt = np.empty(0)
        ba, fo = 0, 0


        for i in range(self.N):
            #print(S)
            Si = [S[3 * i + 0], S[3 * i + 1], S[3 * i + 2]]
            Snorm = np.linalg.norm(Si)
            Sib = []
            Sif = []
            if i <= 1:
                for Bi in self.pulse:
                    fo = i + 1
                    Sib = [0, 0, 0]
                    Sif = [self.J * S[3 * fo + 0], self.J * S[3 * fo + 1], self.J * S[3 * fo + 2]]

                    if t > Bi[0] and t <= Bi[1]:
                        B_e = [Sib[0] + Sif[0] + self.B[0] + self.Kx * Si[0] / (Snorm * Snorm) + self.beta * Bi[2][0],
                               Sib[1] + Sif[1] + self.B[1] + self.Ky * Si[1] / (Snorm * Snorm) + self.beta * Bi[2][1],
                               Sib[2] + Sif[2] + self.B[2] + self.Kz * Si[2] / (Snorm * Snorm) + self.beta * Bi[2][2]]

                        sc_torque = [
                            self.gamma * (Si[1] * (Bi[2][0] * Si[1] -  Bi[2][1] * Si[0]) - Si[2] * (
                                    Bi[2][2] * Si[0] - Bi[2][0] * Si[2])), self.gamma * (
                                    Si[2] * ( Bi[2][1] * Si[2] -  Bi[2][2] * Si[1]) - Si[0] * (
                                    Bi[2][0] * Si[1] -  Bi[2][1] * Si[0])), self.gamma * (
                                    Si[0] * ( Bi[2][2] * Si[0] - Bi[2][0] * Si[2]) - Si[1] * (
                                    Bi[2][1] * Si[2] -  Bi[2][2] * Si[1]))]
                        break

                    else:
                        B_e = [Sib[0] + Sif[0] + self.B[0] + self.Kx * Si[0] / (Snorm * Snorm), Sib[1] + Sif[1] + self.B[1] + self.Ky * Si[1]/(Snorm * Snorm), Sib[2] + Sif[2] + self.B[2] + self.Kz * Si[2]/(Snorm * Snorm)]
                        sc_torque = [0,0,0]

                dSixdt = - self.gamma * (B_e[2] * Si[1] - B_e[1] * Si[2]) - sc_torque[0] - (self.alpha / Snorm) * (
                            Si[1] * (self.gamma * (B_e[1] * Si[0] - B_e[0] * Si[1])) - Si[2] *
                                self.gamma * (B_e[0] * Si[2] - B_e[2] * Si[0]))
                dSiydt = - self.gamma * (B_e[0] * Si[2] - B_e[2] * Si[0]) - sc_torque[1] - (self.alpha / Snorm) * (
                            Si[2] * (self.gamma * (B_e[2] * Si[1] - B_e[1] * Si[2])) - Si[0] *
                                self.gamma * (B_e[1] * Si[0] - B_e[0] * Si[1]) )
                dSizdt = - self.gamma * (B_e[1] * Si[0] - B_e[0] * Si[1]) - sc_torque[2] - (self.alpha / Snorm) * (
                            Si[0] * self.gamma * (B_e[0] * Si[2] - B_e[2] * Si[0]) - Si[1] *
                                self.gamma * (B_e[2] * Si[1] - B_e[1] * Si[2]))
                #print(i, t, [dSixdt, dSiydt, dSizdt])


                dSdt = np.append(dSdt, [dSixdt, dSiydt, dSizdt])

            elif i == self.N - 1:
                a = self.alpha
                self.alpha = 0.01
                ba = i - 1
                Sib = [self.J * S[3 * ba + 0], self.J * S[3 * ba + 1], self.J * S[3 * ba + 2]]
                Sif = [0, 0, 0]
                B_e = [Sib[0] + Sif[0] + self.B[0] + self.Kx * Si[0] / (Snorm * Snorm),
                       Sib[1] + Sif[1] + self.B[1] + self.Ky * Si[1] / (Snorm * Snorm),
                       Sib[2] + Sif[2] + self.B[2] + self.Kz * Si[2] / (Snorm * Snorm)]



                dSixdt = - self.gamma * (B_e[2] * Si[1] - B_e[1] * Si[2]) - (self.alpha / Snorm) * (
                        Si[1] * (self.gamma * (B_e[1] * Si[0] - B_e[0] * Si[1])) - Si[2] *
                        self.gamma * (B_e[0] * Si[2] - B_e[2] * Si[0]))
                dSiydt = - self.gamma * (B_e[0] * Si[2] - B_e[2] * Si[0]) - (self.alpha / Snorm) * (
                        Si[2] * (self.gamma * (B_e[2] * Si[1] - B_e[1] * Si[2])) - Si[0] *
                        self.gamma * (B_e[1] * Si[0] - B_e[0] * Si[1]))
                dSizdt = - self.gamma * (B_e[1] * Si[0] - B_e[0] * Si[1]) - (self.alpha / Snorm) * (
                        Si[0] * self.gamma * (B_e[0] * Si[2] - B_e[2] * Si[0]) - Si[1] *
                        self.gamma * (B_e[2] * Si[1] - B_e[1] * Si[2]))

                dSdt = np.append(dSdt, [dSixdt, dSiydt, dSizdt])
                self.alpha = a


            else:
                fo = i + 1
                ba = i - 1
                Sib = [self.J * S[3 * ba + 0], self.J * S[3 * ba + 1], self.J * S[3 * ba + 2]]
                Sif = [self.J * S[3 * fo + 0], self.J * S[3 * fo + 1], self.J * S[3 * fo + 2]]
                #B_e = [Sib[0] + Sif[0] + self.B[0] + self.Kx * S[0] / (Snorm * Snorm), Sib[1] + Sif[1] + self.B[1] + self.Ky * S[1]/(Snorm * Snorm), Sib[2] + Sif[2] + self.B[2] + self.Kz * S[2]/(Snorm * Snorm)]
                B_e = [Sib[0] + Sif[0] + self.B[0] + self.Kx * Si[0] / (Snorm * Snorm),
                       Sib[1] + Sif[1] + self.B[1] + self.Ky * Si[1] / (Snorm * Snorm),
                       Sib[2] + Sif[2] + self.B[2] + self.Kz * Si[2] / (Snorm * Snorm)]
                #print(B_e)

                dSixdt = - self.gamma * (B_e[2] * Si[1] - B_e[1] * Si[2]) - (self.alpha / Snorm) * (
                        Si[1] * (self.gamma * (B_e[1] * Si[0] - B_e[0] * Si[1])) - Si[2] *
                        self.gamma * (B_e[0] * Si[2] - B_e[2] * Si[0]))
                dSiydt = - self.gamma * (B_e[0] * Si[2] - B_e[2] * Si[0])  - (self.alpha / Snorm) * (
                        Si[2] * (self.gamma * (B_e[2] * Si[1] - B_e[1] * Si[2])) - Si[0] *
                        self.gamma * (B_e[1] * Si[0] - B_e[0] * Si[1]))
                dSizdt = - self.gamma * (B_e[1] * Si[0] - B_e[0] * Si[1])  - (self.alpha / Snorm) * (
                        Si[0] * self.gamma * (B_e[0] * Si[2] - B_e[2] * Si[0]) - Si[1] *
                        self.gamma * (B_e[2] * Si[1] - B_e[1] * Si[2]))

                dSdt = np.append(dSdt, [dSixdt, dSiydt, dSizdt])
                #print(i, t, [dSixdt, dSiydt, dSizdt])
        #print("fat",t,dSdt)
        #dSdt = np.reshape(dSdt, (1, -1))
        #print(dSdt)

        return dSdt


    # S0 = np.array([[1,0,0],[0,0,1],[0,0,1]])



    def get_spin_vec(self,t, i):
        Ox, Oy, Oz = 0, 3 * i, 0
        x = self.S[i*3][t]
        y = self.S[i*3 + 1][t]
        z = self.S[i*3 + 2][t]
        return Ox, Oy, Oz, x, y, z





    def update(self,t):
        for i in range(self.N):
            self.quiver_dic[i].remove()
            self.quiver_dic[i] = self.ax.quiver(*self.get_spin_vec(t, i))

    def doit(self):
        self.S0 = np.reshape(self.S0, (1, -1))
        self.S0 = list(self.S0[0])
        self.fig, self.ax = plt.subplots(subplot_kw=dict(projection="3d"))

        self.Sol = sc.integrate.solve_ivp(self.func_S, self.t, self.S0, t_eval=self.t_eval)
        self.S = self.Sol.y
        self.spin_log = np.reshape(self.S, (self.N,3, -1))
        frames = []

        #self.S = np.reshape(self.S, (-1, self.N, 3))

        for i in range(self.N):
            self.quiver_dic[i] = self.ax.quiver(*self.get_spin_vec(0, i))

        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 300)
        self.ax.set_zlim(-2, 2)
        #self.ax.view_init(elev=90)

        ani = FuncAnimation(self.fig, self.update, frames=len(self.Sol.t), interval=1)
        #ani.save("reverse_spin.gif",writer='imagemagick')
        plt.show()


    def make_i2D_graph(self,i,ax):      #二次元のぐらふを作る。iはi番目のスピン、axはどの軸をとるか
        self.S0 = np.reshape(self.S0, (1, -1))
        self.S0 = list(self.S0[0])
        #self.fig, self.ax = plt.subplots(subplot_kw=dict(projection="3d"))
        print(self.S0)

        self.Sol = sc.integrate.solve_ivp(self.func_S, self.t, self.S0, t_eval=self.t_eval)
        self.S = self.Sol.y
        self.spin_log = np.reshape(self.S, (self.N, 3, -1))
        t = self.t_eval
        y = self.spin_log[i][ax]

        plt.plot(t,y,label= f"{i}S{ax}")

        plt.show()

        N = len(y)   #サンプル数
        f_s = 40000  # サンプリングレート f_s[Hz] (任意)
        dt = 1 / f_s  # サンプリング周期 dt[s]

        y_fft = np.fft.fft(y)  # 離散フーリエ変換
        freq = np.fft.fftfreq(N, d=dt)  # 周波数を割り当てる（※後述）
        Amp = abs(y_fft / (N / 2))  # 音の大きさ（振幅の大きさ）
        plt.plot(freq[1:int(N / 2)], Amp[1:int(N / 2)])  # A-f グラフのプロット
        plt.xscale("log")  # 横軸を対数軸にセット

        plt.show()






if __name__ == '__main__':
    n = 100
    S0 = np.zeros((n, 3))

    for i in range(n):
        S0[i][0] = 1
        S0[i][2] = 0.3


    t = [0,10]  # t(時間)が0〜100まで動き、その時のfを求める。
    t_eval = np.linspace(*t, 2000)

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
        tempul = [1 + i, 1 + i +0.2, [0, -Hs, 0]]
        pulse.append(tempul)


    spin = Spin_wave(0.001, gamma, [0, 0, mu_0 * 12], S0, t, t_eval, mu_0 * 4,0,- mu_0 * 41.6, 0, pulse, n, 3)
    spin.doit()
    spin.make_i2D_graph(50,2)
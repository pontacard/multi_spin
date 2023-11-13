import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import ArtistAnimation
from matplotlib.animation import FuncAnimation
from sin_spin_wave import Sin_Spin_wave

class FMR(Sin_Spin_wave):
    def __init__(self,alpha,edge_alpha,gamma,B,S0,t,t_eval,Kx,Ky,Kz,beta,Amp,omega,N,J,start,stop):
        super().__init__(alpha,edge_alpha,gamma,B,S0,t,t_eval,Kx,Ky,Kz,beta,Amp,omega,N,J,start,stop)



    def func_S(self,t,S):
        dSdt = np.empty(0)
        ba, fo = 0, 0



        for i in range(self.N):
            #print(S)
            Si = [S[3 * i + 0], S[3 * i + 1], S[3 * i + 2]]
            Snorm = np.linalg.norm(Si)
            if i == 0:
                fo = i + 1
                Sib = [0, 0, 0]
                Sif = [self.J * S[3 * fo + 0], self.J * S[3 * fo + 1], self.J * S[3 * fo + 2]]

                Bi = [0, 0, 0]


                if t >= self.start and t <= self.stop:
                    B_e = [Sib[0] + Sif[0] + self.B[0] + self.Amp[0] * np.cos(self.omega[0] * t) + self.Kx * Si[0] / (Snorm * Snorm) + self.beta * Bi[0],
                           Sib[1] + Sif[1] + self.B[1] + self.Amp[1] * np.sin(self.omega[1] * t) + self.Ky * Si[1] / (Snorm * Snorm) + self.beta * Bi[1],
                           Sib[2] + Sif[2] + self.B[2] + self.Amp[2] * np.sin(self.omega[2] * t) + self.Kz * Si[2] / (Snorm * Snorm) + self.beta * Bi[2]]

                    sc_torque = [
                        self.gamma * (Si[1] * (Bi[0] * Si[1] - Bi[1] * Si[0]) - Si[2] * (
                                Bi[2] * Si[0] - Bi[0] * Si[2])), self.gamma * (
                                Si[2] * (Bi[1] * Si[2] - Bi[2] * Si[1]) - Si[0] * (
                                Bi[0] * Si[1] - Bi[1] * Si[0])), self.gamma * (
                                Si[0] * (Bi[2] * Si[0] - Bi[0] * Si[2]) - Si[1] * (
                                Bi[1] * Si[2] - Bi[2] * Si[1]))]


                else:
                    B_e = [Sib[0] + Sif[0] + self.B[0] + self.Kx * Si[0] / (Snorm * Snorm),
                           Sib[1] + Sif[1] + self.B[1] + self.Ky * Si[1] / (Snorm * Snorm),
                           Sib[2] + Sif[2] + self.B[2] + self.Kz * Si[2] / (Snorm * Snorm)]
                    sc_torque = [0, 0, 0]

                dSixdt = - self.gamma * (B_e[2] * Si[1] - B_e[1] * Si[2]) - sc_torque[0] - (self.edge_alpha / Snorm) * (
                        Si[1] * (self.gamma * (B_e[1] * Si[0] - B_e[0] * Si[1])) - Si[2] *
                        self.gamma * (B_e[0] * Si[2] - B_e[2] * Si[0]))
                dSiydt = - self.gamma * (B_e[0] * Si[2] - B_e[2] * Si[0]) - sc_torque[1] - (self.edge_alpha / Snorm) * (
                        Si[2] * (self.gamma * (B_e[2] * Si[1] - B_e[1] * Si[2])) - Si[0] *
                        self.gamma * (B_e[1] * Si[0] - B_e[0] * Si[1]))
                dSizdt = - self.gamma * (B_e[1] * Si[0] - B_e[0] * Si[1]) - sc_torque[2] - (self.edge_alpha / Snorm) * (
                        Si[0] * self.gamma * (B_e[0] * Si[2] - B_e[2] * Si[0]) - Si[1] *
                        self.gamma * (B_e[2] * Si[1] - B_e[1] * Si[2]))
                # print(i, t, [dSixdt, dSiydt, dSizdt])

                dSdt = np.append(dSdt, [dSixdt, dSiydt, dSizdt])

            elif i == self.N - 1:
                ba = i - 1
                Sib = [self.J * S[3 * ba + 0], self.J * S[3 * ba + 1], self.J * S[3 * ba + 2]]
                Sif = [0, 0, 0]
                B_e = [Sib[0] + Sif[0] + self.B[0] + self.Kx * Si[0] / (Snorm * Snorm),
                       Sib[1] + Sif[1] + self.B[1] + self.Ky * Si[1] / (Snorm * Snorm),
                       Sib[2] + Sif[2] + self.B[2] + self.Kz * Si[2] / (Snorm * Snorm)]

                dSixdt = - self.gamma * (B_e[2] * Si[1] - B_e[1] * Si[2]) - (self.edge_alpha / Snorm) * (
                        Si[1] * (self.gamma * (B_e[1] * Si[0] - B_e[0] * Si[1])) - Si[2] *
                        self.gamma * (B_e[0] * Si[2] - B_e[2] * Si[0]))
                dSiydt = - self.gamma * (B_e[0] * Si[2] - B_e[2] * Si[0]) - (self.edge_alpha / Snorm) * (
                        Si[2] * (self.gamma * (B_e[2] * Si[1] - B_e[1] * Si[2])) - Si[0] *
                        self.gamma * (B_e[1] * Si[0] - B_e[0] * Si[1]))
                dSizdt = - self.gamma * (B_e[1] * Si[0] - B_e[0] * Si[1]) - (self.edge_alpha / Snorm) * (
                        Si[0] * self.gamma * (B_e[0] * Si[2] - B_e[2] * Si[0]) - Si[1] *
                        self.gamma * (B_e[2] * Si[1] - B_e[1] * Si[2]))

                dSdt = np.append(dSdt, [dSixdt, dSiydt, dSizdt])


            else:
                fo = i + 1
                ba = i - 1
                Sib = [self.J * S[3 * ba + 0], self.J * S[3 * ba + 1], self.J * S[3 * ba + 2]]
                Sif = [self.J * S[3 * fo + 0], self.J * S[3 * fo + 1], self.J * S[3 * fo + 2]]
                # B_e = [Sib[0] + Sif[0] + self.B[0] + self.Kx * S[0] / (Snorm * Snorm), Sib[1] + Sif[1] + self.B[1] + self.Ky * S[1]/(Snorm * Snorm), Sib[2] + Sif[2] + self.B[2] + self.Kz * S[2]/(Snorm * Snorm)]
                B_e = [Sib[0] + Sif[0] + self.B[0] + self.Kx * Si[0] / (Snorm * Snorm),
                       Sib[1] + Sif[1] + self.B[1] + self.Ky * Si[1] / (Snorm * Snorm),
                       Sib[2] + Sif[2] + self.B[2] + self.Kz * Si[2] / (Snorm * Snorm)]
                # print(B_e)

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
                # print(i, t, [dSixdt, dSiydt, dSizdt])
        # print("fat",t,dSdt)
        # dSdt = np.reshape(dSdt, (1, -1))
        # print(dSdt)

        return dSdt

if __name__ == '__main__':
    n = 50
    S0 = np.zeros((n, 3))

    for i in range(n):
        S0[i][0] = 0
        S0[i][2] = 1

    t = [0, 100]  # t(時間)が0〜100まで動き、その時のfを求める。
    t_eval = np.linspace(*t, 20000)

    mu_0 = 1.2
    gamma = 2.8
    h_div_2e = [0.329, -15]
    sta_M = [1.4, 0]  # 飽和磁化(T)で入れる
    theta = [-2.2, -1]
    j = [4.5, 9]
    d = [1.48, -9]
    Hsn = h_div_2e[0] * theta[0] * j[0] / (sta_M[0] * d[0])
    Hso = h_div_2e[1] + theta[1] + j[1] - (sta_M[1] + d[1])
    Hs = Hsn * (10 ** Hso) * 1000 * (mu_0 / 1200000) * mu_0  # 最後の1000はmTにするため
    print(Hs)

    spin = Sin_Spin_wave(0.0001,0.1, gamma, [0, 0, mu_0 * 12], S0, t, t_eval, 0,0,260, 0, [Hs,Hs,0],[77,77,0] ,n, 1,0,100)
    spin.doit()
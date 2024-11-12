import numpy as np
import scipy.constants as spc
import matplotlib.pyplot as plt

def fspl_linear_scale(d, f):
    return ((4.0 * np.pi * f * d) / spc.c)**2.0
def fspl_log_scale(d, f):
    return 20 * np.log10(d) + 20 * np.log10(f) -147.55
## 1
distances = np.linspace(1, 100000, 10000)
#частоты
f1 = 900000000
f2 = 1900000000
f3 = 28000000000

plt.figure(figsize=(10,8))
plt.plot(distances, [fspl_linear_scale(i, f1) for i in distances], label='При f = 900 МГц')
plt.plot(distances, [fspl_linear_scale(i, f2) for i in distances], label='При f = 1.9 ГГц')
plt.plot(distances, [fspl_linear_scale(i, f3) for i in distances], label='При f = 28 ГГц')
plt.title
plt.xlabel
plt.ylabel
plt.xlim(xmin=distances[0], xmax=distances[-1])
plt.legend()
plt.show()

##2
plt.figure(figsize=(10,8))
plt.plot(distances, [fspl_log_scale(i, f1) for i in distances], label='При f = 900 МГц')
plt.plot(distances, [fspl_log_scale(i, f2) for i in distances], label='При f = 1.9 ГГц')
plt.plot(distances, [fspl_log_scale(i, f3) for i in distances], label='При f = 28 ГГц')
plt.title
plt.xlabel
plt.ylabel
plt.xlim(xmin=distances[0], xmax=distances[-1])
plt.legend()
plt.show()

def P_r_calc(d, f, P_t, G_t, G_r):
    c = spc.c
    tmp1 = (c / (4 * np.pi * d * f))**2
    tmp2 = G_t * G_r * P_t
    tmp3 = tmp1 * tmp2
    return 10 * np.log10(tmp3)

def calc_max_d(distances, f, L, P_t, G_t, G_r):
    d = 0
    for i in distances:
        if P_r_calc(i, f, P_t, G_t, G_r) >= L:
            d = i
        else:
            break
    return d

P_t = 23
G_r = G_t = 10
L = -70

print("Максимальная длина для f = 900МГц: {}".format(calc_max_d(distances, f1, L, P_t, G_t, G_r)))
print("Максимальная длина для f = 1.9ГГц: {}".format(calc_max_d(distances, f2, L, P_t, G_t, G_r)))
print("Максимальная длина для f = 28ГГц: {}".format(calc_max_d(distances, f3, L, P_t, G_t, G_r)))

##3
def UMA_LOS(d, fc):
    PL = 0
    d_bp = 1000 #1 км
    h_bs = 25
    h_ut = 10
    if 10 <= d <= d_bp:
        PL = 28.0 + 22*np.log10(d) + 20*np.log10(fc)
    else:
        PL = 28.0 + 40*np.log10(d) + 20*np.log10(fc) - 9*np.log10(d_bp**2 + (h_bs - h_ut)**2)
    return PL

distances = np.linspace(1, 5000, 1000)
f1 = 900000000
f2 = 1900000000
f3 = 28000000000

plt.figure(figsize=(10,8))
plt.plot(distances, [UMA_LOS(i, f1) for i in distances], label='При f = 900 МГц')
plt.plot(distances, [UMA_LOS(i, f2) for i in distances], label='При f = 1.9 ГГц')
plt.plot(distances, [UMA_LOS(i, f3) for i in distances], label='При f = 28 ГГц')
plt.title
plt.xlabel
plt.ylabel
plt.xlim(xmin=distances[0], xmax=distances[-1])
plt.legend()
plt.show()

##4
def UMA_NLOS(d, fc):
    h_bs = 25
    h_ut = 10
    PLL = 13.54 + 39.08*np.log10(d) + 20*np.log10(fc) - 0.6*(h_ut - 1.5)
    PL = UMA_LOS(d, fc)
    if PL > PLL:
        return PL
    else:
        return PLL

plt.figure(figsize=(10,8))
plt.plot(distances, [UMA_NLOS(i, f1) for i in distances], label='При f = 900 МГц')
plt.plot(distances, [UMA_NLOS(i, f2) for i in distances], label='При f = 1.9 ГГц')
plt.plot(distances, [UMA_NLOS(i, f3) for i in distances], label='При f = 28 ГГц')
plt.title
plt.xlabel
plt.ylabel
plt.xlim(xmin=distances[0], xmax=distances[-1])
plt.legend()
plt.show()

##5
def InH_LOS(d, fc):
    h_ut = 10
    PL = 32.4 + 17.3*np.log10(d) + 20*np.log10(fc)
    return PL

distances = np.linspace(1, 100, 100)
plt.figure(figsize=(10,8))
plt.plot(distances, [InH_LOS(i, f1) for i in distances], label='При f = 900 МГц')
plt.plot(distances, [InH_LOS(i, f2) for i in distances], label='При f = 1.9 ГГц')
plt.plot(distances, [InH_LOS(i, f3) for i in distances], label='При f = 28 ГГц')
plt.title
plt.xlabel
plt.ylabel
plt.xlim(xmin=distances[0], xmax=distances[-1])
plt.legend()
plt.show()

##6
def InH_NLOS(d, fc):
    h_ut = 10
    PL = InH_LOS(d, fc)
    PLL = 38.3*np.log10(d) + 17.3 + 24.9*np.log10(fc)
    if PL > PLL:
        return PL
    else:
        return PLL
distances = np.linspace(1, 86, 100)
plt.figure(figsize=(10,8))
plt.plot(distances, [InH_NLOS(i, f1) for i in distances], label='При f = 900 МГц')
plt.plot(distances, [InH_NLOS(i, f2) for i in distances], label='При f = 1.9 ГГц')
plt.plot(distances, [InH_NLOS(i, f3) for i in distances], label='При f = 28 ГГц')
plt.title
plt.xlabel
plt.ylabel
plt.xlim(xmin=distances[0], xmax=distances[-1])
plt.legend()
plt.show()

##7
def InH_NLOS_alt(d, fc):
    h_ut = 10
    PL = InH_LOS(d, fc)
    PLL = 32.4 + 20*np.log10(fc) + 31.9*np.log10(d)
    if PL > PLL:
        return PL
    else:
        return PLL

plt.figure(figsize=(10,8))
plt.plot(distances, [InH_NLOS_alt(i, f1) for i in distances], label='При f = 900 МГц')
plt.plot(distances, [InH_NLOS_alt(i, f2) for i in distances], label='При f = 1.9 ГГц')
plt.plot(distances, [InH_NLOS_alt(i, f3) for i in distances], label='При f = 28 ГГц')
plt.title
plt.xlabel
plt.ylabel
plt.xlim(xmin=distances[0], xmax=distances[-1])
plt.legend()
plt.show()


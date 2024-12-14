import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from scipy.constants import speed_of_light
from scipy.integrate import quad


def log_scale(value):
    return 10*np.log10(value)

def lin_scale(value):
    return np.power(10, value / 10)


def wave_length(f):
    return speed_of_light / f


def p_C(angle, rxtx=False):
    result = angle / (2 * np.pi)
    if rxtx:
        result = np.power(result, 2)

    return result


def gain(angle):
    return 2 / (1 - np.cos(angle / 2))


def integrand(r, lambda_b, r_b):
    return np.exp(-lambda_b * 2 * r_b * r) / r


def mean_interference(P_t, G_t, G_r, f, lambda_i, p_c, r_b, R, lambda_b):
    part_1 = (P_t * G_t * G_r * wave_length(f) / (8 * np.pi)) * lambda_i * p_c
    part_2 = quad(integrand, r_b, R, args=(lambda_b, r_b))[0]
    return part_1 * part_2

# начальные значения
alpha = np.linspace(0, 2*np.pi, 1000)
alpha_omni = np.array([2*np.pi]*1000)
alpha_lambda_i = np.pi / 18
lambda_i = np.linspace(0, 1, 1000)
P_t = lin_scale(23)
R = 1000

f1 = 900 * (10**6)
f2 = 1800 * (10**6)
f3 = 28000 * (10**6)
r_b = 0.3
lambda_b = 0.4

# направленности антенны передатчика под определенным углом
interference_angle_tx_900 = mean_interference(
    P_t, gain(alpha),10,
    f1,
    np.mean(lambda_i),
    p_C(alpha), r_b, R, lambda_b)


interference_lambda_tx_900 = mean_interference(
    P_t, gain(alpha_lambda_i),10,
    f1,
    lambda_i,
    p_C(alpha_lambda_i), r_b, R, lambda_b)


interference_angle_tx_1800 = mean_interference(
    P_t, gain(alpha),10,
    f2,
    np.mean(lambda_i),
    p_C(alpha), r_b, R, lambda_b)


interference_lambda_tx_1800 = mean_interference(
    P_t, gain(alpha_lambda_i),10,
    f2,
    lambda_i,
    p_C(alpha_lambda_i), r_b, R, lambda_b)

interference_angle_tx_28000 = mean_interference(
    P_t, gain(alpha),10,
    f3,
    np.mean(lambda_i),
    p_C(alpha), r_b, R, lambda_b)


interference_lambda_tx_28000 = mean_interference(
    P_t, gain(alpha_lambda_i),10,
    f3,
    lambda_i,
    p_C(alpha_lambda_i), r_b, R, lambda_b)

# направленности антенн передатчика и приемника под определенным углом


interference_angle_rxtx_900 = mean_interference(
    P_t, gain(alpha), gain(alpha),
    f1,
    np.mean(lambda_i),
    p_C(alpha, rxtx=True), r_b, R, lambda_b)

interference_lambda_rxtx_900 = mean_interference(
    P_t, gain(alpha_lambda_i), gain(alpha_lambda_i),
    f1,
    lambda_i,
    p_C(alpha_lambda_i, rxtx=True), r_b, R, lambda_b)

interference_angle_rxtx_1800 = mean_interference(
    P_t, gain(alpha), gain(alpha),
    f2,
    np.mean(lambda_i),
    p_C(alpha, rxtx=True), r_b, R, lambda_b)

interference_lambda_rxtx_1800 = mean_interference(
    P_t, gain(alpha_lambda_i), gain(alpha_lambda_i),
    f2,
    lambda_i,
    p_C(alpha_lambda_i, rxtx=True), r_b, R, lambda_b)

interference_angle_rxtx_28000 = mean_interference(
    P_t, gain(alpha), gain(alpha),
    f3,
    np.mean(lambda_i),
    p_C(alpha, rxtx=True), r_b, R, lambda_b)

interference_lambda_rxtx_28000 = mean_interference(
    P_t, gain(alpha_lambda_i), gain(alpha_lambda_i),
    f3,
    lambda_i,
    p_C(alpha_lambda_i, rxtx=True), r_b, R, lambda_b)
# всенаправленность антенн

interference_angle_omni_900 = mean_interference(
    P_t, gain(alpha_omni), gain(alpha_omni),
    f1,
    np.mean(lambda_i),
    p_C(alpha_omni, rxtx=True), r_b, R, lambda_b)

interference_lambda_omni_900 = mean_interference(
    P_t, gain(alpha_omni), gain(alpha_omni),
    f1,
    lambda_i,
    p_C(alpha_omni, rxtx=True), r_b, R, lambda_b)

interference_angle_omni_1800 = mean_interference(
    P_t, gain(alpha_omni), gain(alpha_omni),
    f2,
    np.mean(lambda_i),
    p_C(alpha_omni, rxtx=True), r_b, R, lambda_b)

interference_lambda_omni_1800 = mean_interference(
    P_t, gain(alpha_omni), gain(alpha_omni),
    f2,
    lambda_i,
    p_C(alpha_omni, rxtx=True), r_b, R, lambda_b)

interference_angle_omni_28000 = mean_interference(
    P_t, gain(alpha_omni), gain(alpha_omni),
    f3,
    np.mean(lambda_i),
    p_C(alpha_omni, rxtx=True), r_b, R, lambda_b)

interference_lambda_omni_28000 = mean_interference(
    P_t, gain(alpha_omni), gain(alpha_omni),
    f3,
    lambda_i,
    p_C(alpha_omni, rxtx=True), r_b, R, lambda_b)

# график зависимости от угла напрвленности

plt.figure(figsize=(12,10))
plt.plot(alpha, log_scale(interference_angle_tx_900), label='Направленность антенны передатчика')
plt.plot(alpha, log_scale(interference_angle_rxtx_900), label='Направленность антенны передатчика и приемника')
plt.plot(alpha, log_scale(interference_angle_omni_900), label='Всенаправленность')
plt.ylabel('$E[I]$')
plt.xlabel('угол')
plt.xlim(0, 2*np.pi)
plt.title('Зависимость от угла направленности при 900МГц')
plt.grid()
plt.legend()
plt.plot()
plt.show()

plt.figure(figsize=(12,10))
plt.plot(alpha, log_scale(interference_angle_tx_1800), label='Направленность антенны передатчика')
plt.plot(alpha, log_scale(interference_angle_rxtx_1800), label='Направленность антенны передатчика и приемника')
plt.plot(alpha, log_scale(interference_angle_omni_1800), label='Всенаправленность')
plt.ylabel('$E[I]$')
plt.xlabel('угол')
plt.xlim(0, 2*np.pi)
plt.title('Зависимость от угла направленности при 1800МГц')
plt.grid()
plt.legend()
plt.plot()
plt.show()

plt.figure(figsize=(12,10))
plt.plot(alpha, log_scale(interference_angle_tx_28000), label='Направленность антенны передатчика')
plt.plot(alpha, log_scale(interference_angle_rxtx_28000), label='Направленность антенны передатчика и приемника')
plt.plot(alpha, log_scale(interference_angle_omni_28000), label='Всенаправленность')
plt.ylabel('$E[I]$')
plt.xlabel('угол')
plt.xlim(0, 2*np.pi)
plt.title('Зависимость от угла направленности при 28000МГц')
plt.grid()
plt.legend()
plt.plot()
plt.show()

# график зависимости от плотности интерферирующего устройства

plt.figure(figsize=(12,10))
plt.plot(lambda_i, interference_lambda_tx_900, label='Направленность антенны передатчика')
plt.plot(lambda_i, interference_lambda_rxtx_900, label='Направленность антенны передатчика и приемника')
plt.plot(lambda_i, interference_lambda_omni_900, label='Всенаправленность')
plt.ylabel('$E[I]$')
plt.xlabel('$\lambda_I$')
plt.title('Зависимость от $\lambda_i$ при 900МГЦ')
plt.grid()
plt.legend()
plt.plot()
plt.show()

# график зависимости от плотности интерферирующего устройства

plt.figure(figsize=(12,10))
plt.plot(lambda_i, interference_lambda_tx_1800, label='Направленность антенны передатчика')
plt.plot(lambda_i, interference_lambda_rxtx_1800, label='Направленность антенны передатчика и приемника')
plt.plot(lambda_i, interference_lambda_omni_1800, label='Всенаправленность')
plt.ylabel('$E[I]$')
plt.xlabel('$\lambda_I$')
plt.title('Зависимость от $\lambda_i$ при 1800МГЦ')
plt.grid()
plt.legend()
plt.plot()
plt.show()

# график зависимости от плотности интерферирующего устройства

plt.figure(figsize=(12,10))
plt.plot(lambda_i, interference_lambda_tx_28000, label='Направленность антенны передатчика')
plt.plot(lambda_i, interference_lambda_rxtx_28000, label='Направленность антенны передатчика и приемника')
plt.plot(lambda_i, interference_lambda_omni_28000, label='Всенаправленность')
plt.ylabel('$E[I]$')
plt.xlabel('$\lambda_I$')
plt.title('Зависимость от $\lambda_i$ при 28000МГЦ')
plt.grid()
plt.legend()
plt.plot()
plt.show()

# направленности антенны передатчика под определенным углом

interference_angle_tx_900 = mean_interference(
    P_t, gain(alpha),10,
    f1,
    np.mean(lambda_i),
    p_C(alpha), r_b, R, lambda_b)


interference_lambda_tx_900 = mean_interference(
    P_t, gain(alpha_lambda_i),10,
    f1,
    lambda_i,
    p_C(alpha_lambda_i), r_b, R, lambda_b)


interference_angle_tx_1800 = mean_interference(
    P_t, gain(alpha),10,
    f2,
    np.mean(lambda_i),
    p_C(alpha), r_b, R, lambda_b)


interference_lambda_tx_1800 = mean_interference(
    P_t, gain(alpha_lambda_i),10,
    f2,
    lambda_i,
    p_C(alpha_lambda_i), r_b, R, lambda_b)

interference_angle_tx_28000 = mean_interference(
    P_t, gain(alpha),10,
    f3,
    np.mean(lambda_i),
    p_C(alpha), r_b, R, lambda_b)


interference_lambda_tx_28000 = mean_interference(
    P_t, gain(alpha_lambda_i),10,
    f3,
    lambda_i,
    p_C(alpha_lambda_i), r_b, R, lambda_b)

# направленности антенны передатчика под определенным углом

interference_angle_tx_900 = mean_interference(
    P_t, gain(alpha),10,
    f1,
    np.mean(lambda_i),
    p_C(alpha), r_b, R, lambda_b)


interference_lambda_tx_900 = mean_interference(
    P_t, gain(alpha_lambda_i),10,
    f1,
    lambda_i,
    p_C(alpha_lambda_i), r_b, R, lambda_b)


interference_angle_tx_1800 = mean_interference(
    P_t, gain(alpha),10,
    f2,
    np.mean(lambda_i),
    p_C(alpha), r_b, R, lambda_b)


interference_lambda_tx_1800 = mean_interference(
    P_t, gain(alpha_lambda_i),10,
    f2,
    lambda_i,
    p_C(alpha_lambda_i), r_b, R, lambda_b)

interference_angle_tx_28000 = mean_interference(
    P_t, gain(alpha),10,
    f3,
    np.mean(lambda_i),
    p_C(alpha), r_b, R, lambda_b)


interference_lambda_tx_28000 = mean_interference(
    P_t, gain(alpha_lambda_i),10,
    f3,
    lambda_i,
    p_C(alpha_lambda_i), r_b, R, lambda_b)

# всенаправленность антенн

interference_angle_omni_900 = mean_interference(
    P_t, gain(alpha_omni), gain(alpha_omni),
    f1,
    np.mean(lambda_i),
    p_C(alpha_omni, rxtx=True), r_b, R, lambda_b)

interference_lambda_omni_900 = mean_interference(
    P_t, gain(alpha_omni), gain(alpha_omni),
    f1,
    lambda_i,
    p_C(alpha_omni, rxtx=True), r_b, R, lambda_b)

interference_angle_omni_1800 = mean_interference(
    P_t, gain(alpha_omni), gain(alpha_omni),
    f2,
    np.mean(lambda_i),
    p_C(alpha_omni, rxtx=True), r_b, R, lambda_b)

interference_lambda_omni_1800 = mean_interference(
    P_t, gain(alpha_omni), gain(alpha_omni),
    f2,
    lambda_i,
    p_C(alpha_omni, rxtx=True), r_b, R, lambda_b)

interference_angle_omni_28000 = mean_interference(
    P_t, gain(alpha_omni), gain(alpha_omni),
    f3,
    np.mean(lambda_i),
    p_C(alpha_omni, rxtx=True), r_b, R, lambda_b)

interference_lambda_omni_28000 = mean_interference(
    P_t, gain(alpha_omni), gain(alpha_omni),
    f3,
    lambda_i,
    p_C(alpha_omni, rxtx=True), r_b, R, lambda_b)

# график зависимости от угла напрвленности

plt.figure(figsize=(12,10))
plt.plot(alpha, log_scale(interference_angle_tx_900), label='Направленность антенны передатчика')
plt.plot(alpha, log_scale(interference_angle_rxtx_900), label='Направленность антенны передатчика и приемника')
plt.plot(alpha, log_scale(interference_angle_omni_900), label='Всенаправленность')
plt.ylabel('$E[I]$')
plt.xlabel('угол')
plt.xlim(0, 2*np.pi)
plt.title('Зависимость от угла направленности при 900МГц без блокеров')
plt.grid()
plt.legend()
plt.plot()
plt.show()

# график зависимости от угла напрвленности

plt.figure(figsize=(12,10))
plt.plot(alpha, log_scale(interference_angle_tx_1800), label='Направленность антенны передатчика')
plt.plot(alpha, log_scale(interference_angle_rxtx_1800), label='Направленность антенны передатчика и приемника')
plt.plot(alpha, log_scale(interference_angle_omni_1800), label='Всенаправленность')
plt.ylabel('$E[I]$')
plt.xlabel('угол')
plt.xlim(0, 2*np.pi)
plt.title('Зависимость от угла направленности при 1800МГц без блокеров')
plt.grid()
plt.legend()
plt.plot()
plt.show()

# график зависимости от угла напрвленности

plt.figure(figsize=(12,10))
plt.plot(alpha, log_scale(interference_angle_tx_28000), label='Направленность антенны передатчика')
plt.plot(alpha, log_scale(interference_angle_rxtx_28000), label='Направленность антенны передатчика и приемника')
plt.plot(alpha, log_scale(interference_angle_omni_28000), label='Всенаправленность')
plt.ylabel('$E[I]$')
plt.xlabel('угол')
plt.xlim(0, 2*np.pi)
plt.title('Зависимость от угла направленности при 28000МГц без блокеров')
plt.grid()
plt.legend()
plt.plot()
plt.show()

# график зависимости от плотности интерферирующего устройства

plt.figure(figsize=(12,10))
plt.plot(lambda_i, interference_lambda_tx_900, label='Направленность антенны передатчика')
plt.plot(lambda_i, interference_lambda_rxtx_900, label='Направленность антенны передатчика и приемника')
plt.plot(lambda_i, interference_lambda_omni_900, label='Всенаправленность')
plt.ylabel('$E[I]$')
plt.xlabel('$\lambda_I$')
plt.title('Зависимость от $\lambda_i$ при 900МГЦ без блокеров')
plt.grid()
plt.legend()
plt.plot()
plt.show()

# график зависимости от плотности интерферирующего устройства

plt.figure(figsize=(12,10))
plt.plot(lambda_i, interference_lambda_tx_1800, label='Направленность антенны передатчика')
plt.plot(lambda_i, interference_lambda_rxtx_1800, label='Направленность антенны передатчика и приемника')
plt.plot(lambda_i, interference_lambda_omni_1800, label='Всенаправленность')
plt.ylabel('$E[I]$')
plt.xlabel('$\lambda_I$')
plt.title('Зависимость от $\lambda_i$ при 1800МГЦ без блокеров')
plt.grid()
plt.legend()
plt.plot()
plt.show()

# график зависимости от плотности интерферирующего устройства

plt.figure(figsize=(12,10))
plt.plot(lambda_i, interference_lambda_tx_28000, label='Направленность антенны передатчика')
plt.plot(lambda_i, interference_lambda_rxtx_28000, label='Направленность антенны передатчика и приемника')
plt.plot(lambda_i, interference_lambda_omni_28000, label='Всенаправленность')
plt.ylabel('$E[I]$')
plt.xlabel('$\lambda_I$')
plt.title('Зависимость от $\lambda_i$ при 28000МГЦ без блокеров')
plt.grid()
plt.legend()
plt.plot()
plt.show()


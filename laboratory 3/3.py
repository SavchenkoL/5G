import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.constants import speed_of_light as c

def d_2D_3D(d_2D, h_bs, h_ut):
    return np.sqrt(np.power(d_2D, 2) + np.power(h_bs - h_ut, 2))

def pby(lb, rb, dist, hmb, hu, hb):
    y = d_2D_3D(dist, hmb, hu)
    p = 1 - (np.e) ** (-2*lb*rb*(np.sqrt(y**2 - (hmb-hu)**2)*((hb-hu)/(hmb-hu)) + rb))
    return p

dist = np.linspace(1, 100, 300) # расстояние от 1 метра до 100 метров

hmb = 10 #высоты БС
hu = 1.5 #высота АУ
hb = 1.7 #высота человека
rb = 0.3 #радиус человека
lb = 1  #плотность блокаторов

x = pby(lb,rb,dist,hmb,hu,hb)

fig = plt.figure(dpi=100)
plt.plot(dist, x, label='вероятность блокировки')
plt.xlabel('расстояние')
plt.ylabel('вероятность')
plt.legend()
plt.show()

# высота БС 7м, АПУ - 0.5

hmb = 7 #высоты БС
hu = 0.5 #высота АУ
hb = 1.7 #высота человека
rb = 0.3 #радиус человека
lb = 1  #плотность блокаторов

x = pby(lb,rb,dist,hmb,hu,hb)

fig = plt.figure(dpi=100)
plt.plot(dist, x, label='вероятность блокировки')
plt.xlabel('расстояние')
plt.ylabel('вероятность')
plt.legend()
plt.show()


def pby_d2d(lb, rb, y):
    return 1 - (np.e) ** (-2 * lb * rb * (np.sqrt(y ** 2) + rb))

rb = 0.3 #радиус человека
lb = 1  #плотность блокаторов

dist = np.linspace(1, 20, 300)
x = pby_d2d(lb,rb,dist)

fig = plt.figure(dpi=100)
plt.plot(dist, x, label='вероятность блокировки')
plt.xlabel('расстояние')
plt.ylabel('вероятность')
plt.legend()
plt.show()
rb = 0.3 #радиус человека
lb = 1  #плотность блокаторов

dist = np.linspace(0.1, 2, 300)
x = pby_d2d(lb,rb,dist)

fig = plt.figure(dpi=100)
plt.plot(dist, x, label='вероятность блокировки')
plt.xlabel('расстояние')
plt.ylabel('вероятность')
plt.legend()
plt.show()

hmb = 10 #высоты БС
hu = 1.5 #высота АУ
hb = 1.7 #высота человека
rb = 0.3 #радиус человека
lb = 10  #плотность блокаторов

dist = np.linspace(1, 20, 100)
x = pby(lb,rb,dist,hmb,hu,hb)

fig = plt.figure(dpi=100)
plt.plot(dist, x, label='вероятность блокировки')
plt.xlabel('расстояние')
plt.ylabel('вероятность')
plt.legend()
plt.show()

def f_h(x):
    return 1/np.sqrt(2*np.pi) * np.power(np.e, -0.5 * (-x)**2) - 1

k = integrate.quad(f_h, 0, 0.5)
dist = np.linspace(1, 100, 100)

x = 1 - np.e**(np.multiply(k[0], np.multiply(0.5,dist)))

fig = plt.figure(dpi=100)
plt.plot(dist, x, label='вероятность блокировки')
plt.xlabel('расстояние')
plt.ylabel('вероятность')
plt.legend()
plt.show()

def f_h(x):
    return 1.7*x - 1

k = integrate.quad(f_h, 0, 0.5)

x2 = (1 - np.e**(np.multiply(k[0], np.multiply(0.5,dist))))

fig = plt.figure(dpi=100)
plt.plot(dist, x, label='вероятность блокировки')
plt.xlabel('расстояние')
plt.ylabel('вероятность')
plt.legend()
plt.show()

fig = plt.figure(dpi=100)
plt.plot(dist, x, label='вероятность блокировки F(x)')
plt.plot(dist, x2, label='вероятность блокировки 1.7')
plt.xlabel('расстояние')
plt.ylabel('вероятность')
plt.legend()
plt.show()

def logarythm(value):
    # из линейного в лог
    return 10 * np.log10(value)

def linear(value):
    # из лог в линейное
    return np.power(10, value / 10)

f = 28*(10**9) # частоты
dist = np.array([ i for i in range(30, 101)]) # расстояние от 1 метра до 100 метров
P_t = 23 # мощность в дБм передатчика
G_t, G_r = 10, 10 # усиление дБ на приеме и передаче
N_0 = -174 # тепловой шум в дБм
B = 20*(10**6) # полоса частот

def fspl(d, f, h_bs, h_ut):
    loss = np.power((4*np.pi*d*f)/c, 2) # в линейном
    loss = logarythm(loss) # по дефолту переводим в логарифм, но если будет нужно, можем и лин взять
    return loss

def prd(G_r, G_t, P_t, d, f, N_0, B, loss=fspl,  h_bs=None, h_ut=None):
    power_on_reciever = logarythm(linear(P_t) * G_t * G_r) - loss(d, f, h_bs, h_ut) # приемник, для будущего отсечения по -70
    power_on_reciever_b = power_on_reciever - 20
    snr = power_on_reciever - logarythm(B*linear(N_0)) # собственно, получаемая мощность
    snr_b = power_on_reciever_b - logarythm(B*linear(N_0))
    return power_on_reciever, snr, power_on_reciever_b, snr_b

pwr, snr, pwr_b, snr_b = prd(G_r, G_t, P_t, dist, f, N_0, B)

hmb = [i for i in range(5, 41, 1)] #высоты БС
hu = 1.5 #высота АУ
hb = 1.7 #высота человека
rb = 0.3 #радиус человека
lb = 1  #плотность блокаторов

probs = dict.fromkeys(hmb)

for hm in hmb:
    x = pby(lb,rb,dist,hm,hu,hb)
    probs[hm] = x

weighted_snr = dict.fromkeys(hmb)
for hm in hmb:
    prob = probs[hm]
    w_s = []
    for i in range(len(dist)):
        #if  math.isnan(prob[i]):
        #    w_s.append(snr[i])
        #else:
        w_s.append((1 - prob[i]) * snr[i] + prob[i] * snr_b[i]) # Вычисляем взвешенное значение СНР
    weighted_snr[hm] = w_s

fig = plt.figure(dpi=100)
plt.plot(dist, snr, label="без блокатора")
plt.plot(dist, snr_b, label="c блокатором")
plt.plot(dist, weighted_snr[20], label="взвешенное при 20м")
plt.plot(dist, weighted_snr[30], label="взвешенное при 40м")
plt.xlabel('dist')
plt.ylabel('snr')
plt.legend()
plt.show()

f = 28*(10**9) # частоты
dist = 100 # расстояние
P_t = 23 # мощность в дБм передатчика
G_t, G_r = 10, 10 # усиление дБ на приеме и передаче
N_0 = -174 # тепловой шум в дБм
B = 20*(10**6) # полоса частот

pwr, snr, pwr_b, snr_b = prd(G_r, G_t, P_t, dist, f, N_0, B)

hmb = [i for i in range(2, 501, 10)]#np.linspace(2, 20, 50) #высоты БС
hu = 1.5 #высота АУ
hb = 1.7 #высота человека
rb = 0.3 #радиус человека
lb = 1  #плотность блокаторов

probs = []
for h in hmb:
    probs.append(pby(lb,rb,dist,h,hu,hb))

pwr, snr, pwr_b, snr_b

weighted_snr = []
for p in probs:
    w_s = ((1 - p)*snr + p*snr_b) # Вычисляем взвешенное значение СНР
    weighted_snr.append(w_s)

fig = plt.figure(dpi=100)
plt.plot(hmb, weighted_snr, label="SNR(h)")
plt.xlabel('h')
plt.ylabel('snr')
plt.legend()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.constants import speed_of_light
from scipy.integrate import quad

def to_lin(v):
    return 10**(v/10.0)
def to_log(v):
    return 10 * np.log10(v)

f1 = 0.9*(10**9)
f2 = 1.8*(10**9)
f3 = 28*(10**9)
P_t = to_lin(23)
G_t = to_lin(10)
G_r = to_lin(10)
N = to_lin(-174)
B = 20*(10**6)
P_n = N * B


def fspl(y, f):
    a = (4 * np.pi * f / speed_of_light)**2.0
    b = (400 * np.pi * f / speed_of_light)**2.0
    if a <= y <= b:
        return speed_of_light / (792 * np.pi * f * np.sqrt(y))
    else:
        return 0


a = (4 * np.pi * f1 / speed_of_light)**2.0
b = (400 * np.pi * f1 / speed_of_light)**2.0
X_1 = np.linspace(a, b, 10000)
Y_1 = [fspl(i, f1) for i in X_1]


a = (4 * np.pi * f2 / speed_of_light)**2.0
b = (400 * np.pi * f2 / speed_of_light)**2.0
X_2 = np.linspace(a, b, 10000)
Y_2 = [fspl(i, f2) for i in X_2]

a = (4 * np.pi * f3 / speed_of_light)**2.0
b = (400 * np.pi * f3 / speed_of_light)**2.0
X_3 = np.linspace(a, b, 10000)
Y_3 = [fspl(i, f3) for i in X_3]

plt.figure(figsize=(12,10))
plt.plot([to_log(i) for i in X_1], Y_1, label='При f = 900 МГц')
plt.plot([to_log(i) for i in X_2], Y_2, label='При f = 1.8 ГГц')
plt.plot([to_log(i) for i in X_3], Y_3, label='При f = 28 ГГц')
plt.title("Плотность распределения функции FSPL для равномерного распределения")
plt.xlabel("FSPL")
plt.ylabel("f(x)")
#plt.xlim(xmin=to_log(a), xmax=to_log(b))
plt.legend()
plt.show()

def fspl_exp(y, f):
    if y > 0 :
        return 0.02*np.e**(-0.02*speed_of_light*np.sqrt(y)/(4*np.pi*f))*abs(speed_of_light/(8*np.pi*f*np.sqrt(y)))
    else:
        return 0
a = 1
b = 100
X = np.linspace(a, b, 1000)
Y_1 = [fspl_exp(i, f1) for i in X]
Y_2 = [fspl_exp(i, f2) for i in X]
Y_3 = [fspl_exp(i, f3) for i in X]


plt.figure(figsize=(12,10))
plt.plot([to_log(i) for i in X], Y_1, label='При f = 900 МГц')
plt.plot([to_log(i) for i in X], Y_2, label='При f = 1.8 ГГц')
plt.plot([to_log(i) for i in X], Y_3, label='При f = 28 ГГц')
plt.title("Плотность распределения функции FSPL для экспоненциального распределения")
plt.xlabel("FSPL")
plt.ylabel("f(x)")
plt.xlim(xmin=to_log(a), xmax=to_log(b))
plt.legend()
plt.show()

def P_r(y, P_t, G_t, G_r, f):
    a = P_t * G_t * G_r * ((speed_of_light / (400 * np.pi * f))**2)
    b = P_t * G_t * G_r * ((speed_of_light / (4 * np.pi * f))**2)
    if a <= y <= b:
        return np.sqrt(P_t * G_t * G_r) * speed_of_light / (792 * np.pi * f * y * np.sqrt(y))
    else:
        return 0

a = P_t * G_t * G_r * ((speed_of_light / (400 * np.pi * f1))**2)
b = P_t * G_t * G_r * ((speed_of_light / (4 * np.pi * f1))**2)
X_1 = np.linspace(a, b, 10000)
Y_1 = [P_r(i, P_t, G_t, G_r, f1) for i in X_1]

a = P_t * G_t * G_r * ((speed_of_light / (400 * np.pi * f2))**2)
b = P_t * G_t * G_r * ((speed_of_light / (4 * np.pi * f2))**2)
X_2 = np.linspace(a, b, 10000)
Y_2 = [P_r(i, P_t, G_t, G_r, f2) for i in X_2]

a = P_t * G_t * G_r * ((speed_of_light / (400 * np.pi * f3))**2)
b = P_t * G_t * G_r * ((speed_of_light / (4 * np.pi * f3))**2)
X_3 = np.linspace(a, b, 10000)
Y_3 = [P_r(i, P_t, G_t, G_r, f3) for i in X_3]

plt.figure(figsize=(12, 10))
plt.plot([to_log(i) for i in X_1], Y_1, label='При f = 900 МГц')
plt.plot([to_log(i) for i in X_2], Y_2, label='При f = 1.8 ГГц')
plt.plot([to_log(i) for i in X_3], Y_3, label='При f = 28 ГГц')
plt.title("Плотность распределения функции P_r для равномерного распределения")
plt.xlabel("P_r")
plt.ylabel("f(x)")
plt.xlim(xmin=to_log(a), xmax=to_log(b))
plt.legend()
plt.show()

plt.figure(figsize=(12, 10))
plt.plot([to_log(i) for i in X_1], Y_1, label='При f = 900 МГц')
plt.plot([to_log(i) for i in X_2], Y_2, label='При f = 1.8 ГГц')
plt.title("Плотность распределения функции P_r для равномерного распределения")
plt.xlabel("P_r")
plt.ylabel("f(x)")
plt.xlim(xmin=to_log(a), xmax=to_log(b))
plt.legend()
plt.show()

def SNR(y, P_t, G_t, G_r, f, P_n):
    a = (P_t * G_t * G_r / P_n) * ((speed_of_light / (400 * np.pi * f))**2)
    b = (P_t * G_t * G_r / P_n) * ((speed_of_light / (4 * np.pi * f))**2)
    if a <= y <= b:
        return np.sqrt(P_t * G_t * G_r) * speed_of_light / (792 * np.pi * f * y * np.sqrt(y * P_n))
    else:
        return 0

a = (P_t * G_t * G_r / P_n) * ((speed_of_light / (400 * np.pi * f1))**2)
b = (P_t * G_t * G_r / P_n) * ((speed_of_light / (4 * np.pi * f1))**2)
X_1 = np.linspace(a, b, 10000)
Y_1 = [SNR(i, P_t, G_t, G_r, f1, P_n) for i in X_1]

a = (P_t * G_t * G_r / P_n) * ((speed_of_light / (400 * np.pi * f2))**2)
b = (P_t * G_t * G_r / P_n) * ((speed_of_light / (4 * np.pi * f2))**2)
X_2 = np.linspace(a, b, 10000)
Y_2 = [SNR(i, P_t, G_t, G_r, f2, P_n) for i in X_2]

a = (P_t * G_t * G_r / P_n) * ((speed_of_light / (400 * np.pi * f3))**2)
b = (P_t * G_t * G_r / P_n) * ((speed_of_light / (4 * np.pi * f3))**2)
X_3 = np.linspace(a, b, 10000)
Y_3 = [SNR(i, P_t, G_t, G_r, f3, P_n) for i in X_3]

plt.figure(figsize=(12,10))
plt.plot([to_log(i) for i in X_1], Y_1, label='При f = 900 МГц')
plt.plot([to_log(i) for i in X_2], Y_2, label='При f = 1.8 ГГц')
plt.plot([to_log(i) for i in X_3], Y_3, label='При f = 28 ГГц')
plt.title("Плотность распределения функции SNR для равномерного распределения")
plt.xlabel("SNR (дБ)")
plt.ylabel("f(x)")
plt.legend()
plt.show()

plt.figure(figsize=(12,10))
plt.plot([to_log(i) for i in X_1], Y_1, label='При f = 900 МГц')
plt.plot([to_log(i) for i in X_2], Y_2, label='При f = 1.8 ГГц')
plt.title("Плотность распределения функции SNR для равномерного распределения")
plt.xlabel("SNR (дБ)")
plt.ylabel("f(x)")
plt.legend()
plt.show()

def shann(y, P_t, G_t, G_r, f, B, P_n):
    a = B * np.log2((P_t * G_t * G_r / P_n) * ((speed_of_light / (400 * np.pi * f))**2) + 1)
    b = B * np.log2((P_t * G_t * G_r / P_n) * ((speed_of_light / (4 * np.pi * f))**2) + 1)
    if a <= y <= b:
        return np.sqrt(P_t * G_t * G_r) * speed_of_light * np.log(2) * (2.0**(y/B)) / (792 * np.pi * f * (2.0**(y/B) - 1) * np.sqrt(2.0**(y/B) - 1)*B)
    else:
        return 0

a = B * np.log2((P_t * G_t * G_r / P_n) * ((speed_of_light / (400 * np.pi * f1))**2) + 1)
b = B * np.log2((P_t * G_t * G_r / P_n) * ((speed_of_light / (4 * np.pi * f1))**2) + 1)
X_1 = np.linspace(a, b, 1000)
Y_1 = [shann(i, P_t, G_t, G_r, f1, B, P_n) for i in X_1]

a = B * np.log2((P_t * G_t * G_r / P_n) * ((speed_of_light / (400 * np.pi * f2))**2) + 1)
b = B * np.log2((P_t * G_t * G_r / P_n) * ((speed_of_light / (4 * np.pi * f2))**2) + 1)
X_2 = np.linspace(a, b, 1000)
Y_2 = [shann(i, P_t, G_t, G_r, f2, B, P_n) for i in X_2]

a = B * np.log2((P_t * G_t * G_r / P_n) * ((speed_of_light / (400 * np.pi * f3))**2) + 1)
b = B * np.log2((P_t * G_t * G_r / P_n) * ((speed_of_light / (4 * np.pi * f3))**2) + 1)
X_3 = np.linspace(a, b, 1000)
Y_3 = [shann(i, P_t, G_t, G_r, f3, B, P_n) for i in X_3]

plt.figure(figsize=(12,10))
plt.plot([to_log(i) for i in X_1], Y_1, label='При f = 900 МГц')
plt.plot([to_log(i) for i in X_2], Y_2, label='При f = 1.8 ГГц')
plt.plot([to_log(i) for i in X_3], Y_3, label='При f = 28 ГГц')
plt.title("Плотность распределения функции скорости Шеннона для равномерного распределения")
plt.xlabel("Скорость Шеннона")
plt.ylabel("f(x)")
plt.legend()
plt.show()


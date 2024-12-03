import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def G1(alpha):
    return 2/(1 - np.cos(alpha/2))

def G2(alpha, k):
    g1 = 2 * np.power( (1 - np.cos(alpha/2)) + k*(1 + np.cos(alpha/2)), -1)
    g2 = k*g1
    return g1,g2

def to_log_scale(v):
    return 10 * np.log10(v)

angles = np.arange(-np.pi,np.pi,np.pi/8)
angles2 = np.arange(2*-np.pi,2*np.pi,np.pi/36)

fig1 = plt.figure(dpi=120)
plt.polar(angles, G1(angles), label='конус без потери')
plt.title('Модель без потерь')
plt.legend()
plt.show()

plt.figure(dpi=120)
plt.polar(angles2, G2(angles2,0.01)[0], label='главный лепесток')
plt.polar(angles2, G2(angles2,0.01)[1], label='боковые лепестки')
plt.title('Модель с потерями: k=0.01')
plt.legend()
plt.show()

plt.figure(dpi=120)
plt.polar(angles2, G2(angles2,0.1)[0], label='главный лепесток')
plt.polar(angles2, G2(angles2,0.1)[1], label='боковые лепестки')
plt.title('Модель с потерями: k=0.1')
plt.legend()
plt.show()

plt.figure(dpi=120)
plt.polar(angles2, G2(angles2,0.2)[0], label='главный лепесток')
plt.polar(angles2, G2(angles2,0.2)[1], label='боковые лепестки')
plt.title('Модель с потерями: k=0.2')
plt.legend()
plt.show()

alphas = np.linspace(0.01, 2 * np.pi, 1000)
k1 = 0.01
k2 = 0.1
k3 = 0.2
G = []
G1_k1 = []
G1_k2 = []
G1_k3 = []
G2_k1 = []
G2_k2 = []
G2_k3 = []
for i in alphas:
    G.append(G1(i))
    G1_k1.append(G2(i, k1)[0])
    G1_k2.append(G2(i, k2)[0])
    G1_k3.append(G2(i, k3)[0])
    G2_k1.append(G2(i, k1)[1])
    G2_k2.append(G2(i, k2)[1])
    G2_k3.append(G2(i, k3)[1])

plt.figure(figsize=(10, 8))
plt.plot(alphas, [to_log_scale(i) for i in G], label='Модель без потерь')
plt.plot(alphas, [to_log_scale(i) for i in G1_k1], label=f'потери на главном лепестке, k={k1}')
plt.plot(alphas, [to_log_scale(i) for i in G1_k2], label=f'потери на главном лепестке, k={k2}')
plt.plot(alphas, [to_log_scale(i) for i in G1_k3], label=f'потери на главном лепестке, k={k3}')
plt.plot(alphas, [to_log_scale(i) for i in G2_k1], label=f'потери на заднем и боковых лепестках, k={k1}')
plt.plot(alphas, [to_log_scale(i) for i in G2_k2], label=f'потери на заднем и боковых лепестках, k={k2}')
plt.plot(alphas, [to_log_scale(i) for i in G2_k3], label=f'потери на заднем и боковых лепестках, k={k3}')
plt.title("Изменение G в зависимости от α")
plt.xlabel("α")
plt.ylabel("G")
plt.xlim(xmin=alphas[0], xmax=alphas[-1])
plt.legend()
plt.show()

plt.figure(figsize=(10,8))
plt.plot(alphas, G1_k1, label=f'потери на главном лепестке, k={k1}')
plt.plot(alphas, G1_k2, label=f'потери на главном лепестке, k={k2}')
plt.plot(alphas, G1_k3, label=f'потери на главном лепестке, k={k3}')
plt.title("Изменение G в зависимости от α")
plt.xlabel("α")
plt.ylabel("G")
plt.xlim(xmin=alphas[0], xmax=alphas[-1])
plt.legend()
plt.show()

plt.figure(figsize=(10,8))
plt.plot(alphas, G1_k2, label=f'потери на главном лепестке, k={k2}')
plt.plot(alphas, G1_k3, label=f'потери на главном лепестке, k={k3}')
plt.title("Изменение G в зависимости от α")
plt.xlabel("α")
plt.ylabel("G")
plt.xlim(xmin=alphas[0], xmax=alphas[-1])
plt.legend()
plt.show()

plt.figure(figsize=(10,8))
plt.plot(alphas, G2_k1, label=f'потери на заднем и боковых лепестках, k={k1}')
plt.plot(alphas, G2_k2, label=f'потери на заднем и боковых лепестках, k={k2}')
plt.plot(alphas, G2_k3, label=f'потери на заднем и боковых лепестках, k={k3}')
plt.title("Изменение G в зависимости от α")
plt.xlabel("α")
plt.ylabel("G")
plt.xlim(xmin=alphas[0], xmax=alphas[-1])
plt.legend()
plt.show()

# Задание 2
N = np.arange(1, 129, 1)
G = []
G1_k1 = []
G1_k2 = []
G1_k3 = []
G2_k1 = []
G2_k2 = []
G2_k3 = []
for i in N:
    alpha = np.radians(102 / i)
    G.append(G1(alpha))
    G1_k1.append(G2(alpha, k1)[0])
    G1_k2.append(G2(alpha, k2)[0])
    G1_k3.append(G2(alpha, k3)[0])
    G2_k1.append(G2(alpha, k1)[1])
    G2_k2.append(G2(alpha, k2)[1])
    G2_k3.append(G2(alpha, k3)[1])

plt.figure(figsize=(10,8))
plt.plot(N, [to_log_scale(i) for i in G], label='Модель без потерь')
plt.plot(N, [to_log_scale(i) for i in G1_k1], label=f'потери на главном лепестке, k={k1}')
plt.plot(N, [to_log_scale(i) for i in G1_k2], label=f'потери на главном лепестке, k={k2}')
plt.plot(N, [to_log_scale(i) for i in G1_k3], label=f'потери на главном лепестке, k={k3}')
plt.plot(N, [to_log_scale(i) for i in G2_k1], label=f'потери на заднем и боковых лепестках, k={k1}')
plt.plot(N, [to_log_scale(i) for i in G2_k2], label=f'потери на заднем и боковых лепестках, k={k2}')
plt.plot(N, [to_log_scale(i) for i in G2_k3], label=f'потери на заднем и боковых лепестках, k={k3}')
plt.title("Изменение G в зависимости от N")
plt.xlabel("N")
plt.ylabel("G")
plt.legend()
plt.show()

N = np.arange(1, 25, 1)
G = []
G1_k1 = []
G1_k2 = []
G1_k3 = []
G2_k1 = []
G2_k2 = []
G2_k3 = []
for i in N:
    alpha = np.radians(102 / i)
    G.append(G1(alpha))
    G1_k1.append(G2(alpha, k1)[0])
    G1_k2.append(G2(alpha, k2)[0])
    G1_k3.append(G2(alpha, k3)[0])
    G2_k1.append(G2(alpha, k1)[1])
    G2_k2.append(G2(alpha, k2)[1])
    G2_k3.append(G2(alpha, k3)[1])

plt.figure(figsize=(10,8))
plt.plot(N, G1_k2, label=f'потери на главном лепестке, k={k2}')
plt.plot(N, G1_k3, label=f'потери на главном лепестке, k={k3}')
plt.title("Изменение G в зависимости от N")
plt.xlabel("N")
plt.ylabel("G")
plt.legend()
plt.show()

plt.figure(figsize=(10,8))
plt.plot(N, G2_k1, label=f'потери на заднем и боковых лепестках, k={k1}')
plt.plot(N, G2_k2, label=f'потери на заднем и боковых лепестках, k={k2}')
plt.plot(N, G2_k3, label=f'потери на заднем и боковых лепестках, k={k3}')
plt.title("Изменение G в зависимости от N")
plt.xlabel("N")
plt.ylabel("G")
plt.legend()
plt.show()


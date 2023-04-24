# cost/loss function

import numpy as np
import matplotlib.pyplot as plt

# dataset

x = 4 * np.random.rand(100, 1) #rand 100 satir 1 sutun random sayi
y = 9 + (3 * x) + np.random.rand(100, 1)

# scatterplot

plt.scatter(x,y)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15, rotation=0)
plt.show()

# regresyon katsayilarini hesaplayan analitik fonk

def AnalyticsMethod(x, y):
    ssxy = (x * y).sum() - (len(x) * x.mean() * y.mean())
    ssxx = ((x - x.mean()) ** 2).sum()
    m = ssxy / ssxx
    c = y.mean() - (m * x.mean())
    return [m, c]

m, c = AnalyticsMethod(x, y)

print(f'm(theta 0) degeri: {m}')
print(f'c(theta 1) degeri: {c}')

# m ve c ile dogruyu cizdirelim, scatter  plot uzerinde regresyon dogrusu

plt.scatter(x, y)
plt.plot(x, (x * m + c), color = 'red' )
plt.xlabel('x', fontsize = 15)
plt.ylabel("Y", fontsize = 15, rotation = 0)
plt.show() # noktalara en yakin mesafeden gecen dogruyu bulduk

# Gradient Descent

# maliyet (cost) func

def CostFunction(theta, X, y): # katsayilarin vektoru
    m = len(y)
    c = (1/2 * m) * np.sum(np.square((X.dot(theta)) - y))
    return c

# theta: reg katsayilarinin matrisi
# alpha: learning rate

def GradientDescent(X, y, theta, alpha, iterations):
    m = len(y)
    thetas = np.zeros((iterations, 2))
    costs = np.zeros(iterations)

    for i in range(iterations):
        print(i)
        theta = theta - (1 / m) * alpha * (X.T.dot((X.dot(theta)) - y))
        thetas[i, :] = theta.T
        costs[i] = CostFunction(theta, X, y)
    return theta, thetas, costs

alpha = 0.01
iterations = 3000
theta = np.random.randn(2, 1)

# her bir ozellik icin sabit hata payi eklenir

X_bias = np.c_[np.ones((len(x), 1)), x]

# Gradient Decent func calistirma

theta, thetas, costs = GradientDescent(X_bias, y, theta, alpha, iterations)

print(f'theta 0 (m) degeri: {theta[0][0]}')
print(f"theta 1 değeri: {theta[1][0]}")
print(f"maliyet/cost/MSE(L2 Loss) degeri: {costs[-1]}")

# her bir iteration icin cost func'daki degisimlere bakalim

plt.scatter(iter, costs)
plt.xlabel('Iterations Sayisi')
plt.ylabel('Cost/MSE')
plt.box(None)
plt.show()






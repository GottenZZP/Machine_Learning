# coding:utf8

import numpy as np


# HMM类
class HMM:
    def __init__(self, Ann, Bnm, Pi, O):
        self.A = np.array(Ann, np.float)
        self.B = np.array(Bnm, np.float)
        self.Pi = np.array(Pi, np.float)
        self.O = np.array(O, np.float)
        self.N = self.A.shape[0]
        self.M = self.B.shape[1]


# 维特比算法
def viterbi(self):

    T = len(self.O)
    I = np.zeros(T, np.float)

    delta = np.zeros((T, self.N), np.float)
    psi = np.zeros((T, self.N), np.float)

    for i in range(self.N):
        delta[0, i] = self.Pi[i] * self.B[i, self.O[0]]
        psi[0, i] = 0

    for t in range(1, T):
        for i in range(self.N):
            delta[t, i] = self.B[i, self.O[t]] * np.array([delta[t - 1, j] * self.A[j, i]
                                                           for j in range(self.N)]).max()
            psi[t, i] = np.array([delta[t - 1, j] * self.A[j, i]
                                  for j in range(self.N)]).argmax()

    P_T = delta[T - 1, :].max()
    I[T - 1] = delta[T - 1, :].argmax()

    for t in range(T - 2, -1, -1):
        I[t] = psi[t + 1, I[t + 1]]

    return I


# 前向算法
def forward(self):
    T = len(self.O)
    alpha = np.zeros((T, self.N), np.float)

    for i in range(self.N):
        alpha[0, i] = self.Pi[i] * self.B[i, self.O[0]]

    for t in range(T - 1):
        for i in range(self.N):
            summation = 0
            for j in range(self.N):
                summation += alpha[t, j] * self.A[j, i]
            alpha[t + 1, i] = summation * self.B[i, self.O[t + 1]]

    summation = 0.0
    for i in range(self.N):
        summation += alpha[T - 1, i]
    Polambda = summation
    return Polambda, alpha


# 后向算法
def backward(self):
    T = len(self.O)
    beta = np.zeros((T, self.N), np.float)
    for i in range(self.N):
        beta[T - 1, i] = 1.0

    for t in range(T - 2, -1, -1):
        for i in range(self.N):
            summation = 0.0
            for j in range(self.N):
                summation += self.A[i, j] * self.B[j, self.O[t + 1]] * beta[t + 1, j]
            beta[t, i] = summation

    Polambda = 0.0
    for i in range(self.N):
        Polambda += self.Pi[i] * self.B[i, self.O[0]] * beta[0, i]
    return Polambda, beta


# Baum_Welch算法
def Baum_Welch(self):
    T = len(self.O)
    V = [k for k in range(self.M)]

    self.A = np.array(([[0, 1, 0, 0], [0.4, 0, 0.6, 0], [0, 0.4, 0, 0.6], [0, 0, 0.5, 0.5]]), np.float)
    self.B = np.array(([[0.5, 0.5], [0.3, 0.7], [0.6, 0.4], [0.8, 0.2]]), np.float)

    self.Pi = np.array(([1.0 / self.N] * self.N), np.float)

    x = 1
    delta_lambda = x + 1
    times = 0
    while delta_lambda > x:
        Polambda1, alpha = self.forward()
        Polambda2, beta = self.backward()
        gamma = self.compute_gamma(alpha, beta)
        xi = self.compute_xi(alpha, beta)

        lambda_n = [self.A, self.B, self.Pi]

        for i in range(self.N):
            for j in range(self.N):
                numerator = sum(xi[t, i, j] for t in range(T - 1))
                denominator = sum(gamma[t, i] for t in range(T - 1))
                self.A[i, j] = numerator / denominator

        for j in range(self.N):
            for k in range(self.M):
                numerator = sum(gamma[t, j] for t in range(T) if self.O[t] == V[k])
                denominator = sum(gamma[t, j] for t in range(T))
                self.B[i, k] = numerator / denominator

        for i in range(self.N):
            self.Pi[i] = gamma[0, i]

        delta_A = map(abs, lambda_n[0] - self.A)
        delta_B = map(abs, lambda_n[1] - self.B)
        delta_Pi = map(abs, lambda_n[2] - self.Pi)
        delta_lambda = sum([sum(sum(delta_A)), sum(sum(delta_B)), sum(delta_Pi)])
        times += 1
        print(times)

    return self.A, self.B, self.Pi


A = np.array([[0.5, 0.2, 0.3],
              [0.3, 0.5, 0.2],
              [0.2, 0.3, 0.5]])
B = np.array([[0.5, 0.5],
              [0.4, 0.6],
              [0.7, 0.3]])
Pi = np.array([0.2, 0.4, 0.4])
O = np.array([2.0, 1.0, 2.0])

hmm = HMM(A, B, Pi, O)

Polambda, alpha = forward(hmm)

print(Polambda, alpha)

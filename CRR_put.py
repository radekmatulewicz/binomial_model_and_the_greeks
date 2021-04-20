import numpy as np
from tqdm import tqdm

def CRR(S0, r, sigma, T, K, put_call, M):
    dt = T / M
    U = np.exp(sigma * np.sqrt(dt))
    D = 1 / U
    p = (np.exp(r * dt) - D) / (U - D)
    DF = np.exp(-r * dt)

    S = np.zeros([M + 1, M + 1])

    for i in tqdm(range(M + 1)):
        for j in range(i + 1):
            S[j, i] = S0 * (U ** (i - j)) * (D ** j)

    Ve = np.zeros([M + 1, M + 1])
    Va = np.zeros([M + 1, M + 1])
    Ve[:, M] = np.maximum(np.zeros(M + 1), put_call*(S[:, M] - K))
    Va[:, M] = np.maximum(np.zeros(M + 1), put_call*(S[:, M] - K))

    pom1 = np.zeros(M+1)
    pom2 = np.maximum(np.zeros(M + 1), put_call*(S[:, M] - K))
    for i in tqdm(np.arange(M - 1, -1, -1)):
        for j in np.arange(0, i + 1):
            Ve[j, i] = DF * (p * Ve[j, i + 1] + (1 - p) * Ve[j + 1, i + 1])
            pom1[j] = DF * (p * pom2[j] + (1 - p) * pom2[j + 1])
            Va[j, i] = np.maximum(np.maximum(0, put_call*(S[j, i] - K)), DF * (p * Va[j, i + 1] + (1 - p) * Va[j + 1, i + 1]))
        pom2 = pom1

    # Greckie parametry
    # Delta:
    delta_E = (Ve[0, 1] - Ve[1, 1]) / ((S0 * U) - (S0 * D))
    delta_A = (Va[0, 1] - Va[1, 1]) / ((S0 * U) - (S0 * D))

    # Gamma:
    h = 0.5 * (S[0, 2] - S[2, 2])
    delta1_E = (Ve[1, 2] - Ve[2, 2]) / (S0 - S[2, 2])
    delta2_E = (Ve[0, 2] - Ve[1, 2]) / (S[0, 2] - S0)
    gamma_E = (delta2_E - delta1_E) / h
    delta1_A = (Va[1, 2] - Va[2, 2]) / (S0 - S[2, 2])
    delta2_A = (Va[0, 2] - Va[1, 2]) / (S[0, 2] - S0)
    gamma_A = (delta2_A - delta1_A) / h

    # Theta
    theta_E = (Ve[1, 2] - Ve[0, 0]) / (2 * dt)
    theta_A = (Va[1, 2] - Va[0, 0]) / (2 * dt)

    return Ve, pom2, Va, delta_E, delta_A, gamma_E, gamma_A, theta_E, theta_A
import numpy as np

Rate = np.array([[5, 3, 0, 1],
              [4, 0, 0, 1],
              [1, 1, 0, 5],
              [1, 0, 0, 4],
              [0, 1, 5, 4]])
'''
E = (y -r_pred)^2 = (y - P[i] * Q[j)^2
dE/dP = -2(y - PQ)Q
P = P - lr * dE/dP = P + lr * 2(y - PQ) * Q

'''
def matrix_factorization(Rate, k, epochs, lr):
    num_user, num_item = Rate.shape
    P = np.random.rand(num_user, k)
    Q = np.random.rand(num_item, k)

    for epoch in range(epochs):
        for i in range(num_user):
            for j in range(num_item):
                r_pred = np.dot(P[i], Q[j])
                error = R[i, j] - r_pred

                P[i] += lr * error * Q[j] - reg * P[i] # 正则化
                Q[i] += lr * error * P[i] - reg * Q[j]
        
    # 矩阵形式
    for epoch in range(epochs):
        R_pred = np.matmul(P, Q.T)
        E = R - R_pred
        # loss = np.linalg.norm(R - R_pred)
        loss = np.sum(E ** 2) # (n, m)
        P += lr * (np.dot(E.T, Q) - reg * P)
        Q += lr * (np.dot(E, P) - reg * Q)



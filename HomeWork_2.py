import numpy as np

D, N = 4, 3
Gamma = 0.5

def w1(X, y):
    X_mul_X_t = np.dot(X, np.transpose(X))
    I = np.eye(D)
    XXt_plus_gamma_I = X_mul_X_t + Gamma * I
    result = np.dot(np.dot(np.linalg.inv(XXt_plus_gamma_I), X), y)
    return np.round(result, 5)

def w2(X, y):
    X_t_mul_X = np.dot(np.transpose(X), X)
    I = np.eye(N)
    XtX_plus_gamma_I = X_t_mul_X + Gamma * I
    result = np.dot(np.dot(X, np.linalg.inv(XtX_plus_gamma_I)), y)
    return np.round(result, 5)

# 1. Uniform Distribution
X_uni = np.random.uniform(0, 1, [D, N])
y_uni = np.random.uniform(0, 1, N)

w1_uni = w1(X_uni, y_uni)
w2_uni = w2(X_uni, y_uni)
print("1. Uniform Distribution")
print("w1 :", w1_uni)
print("w2 :", w2_uni)

print("w1 == w2 ? :", np.array_equal(w1_uni, w2_uni), "\n")

#2. Normal Distribution
X_nor = np.random.normal(0, 1, [D, N])
y_nor = np.random.normal(0, 1, N)

w1_nor = w1(X_nor, y_nor)
w2_nor = w2(X_nor, y_nor)
print("2. Normal Distribution")
print("w1 :", w1_nor)
print("w2 :", w2_nor)

print("w1 == w2 ? :", np.array_equal(w1_nor, w2_nor))
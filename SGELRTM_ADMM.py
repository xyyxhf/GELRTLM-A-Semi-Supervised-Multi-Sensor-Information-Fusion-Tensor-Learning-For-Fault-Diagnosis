
import numpy as np
from scipy.optimize import minimize
class SGELRTM_ADMM:
    def __init__(self, Xl, Xu, y, xlnum, L, tau, lambda_, max_iter=100, eps=1e-8, rho=0.01, eta=0.999):
        super().__init__()  
        self.Xl = Xl
        self.Xu = Xu
        self.y = y
        self.labelednum = xlnum
        self.L = L
        self.tau = tau
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.eps = eps
        self.rho = rho
        self.eta = eta
        
        self.W = None
        self.W_iter = []
        self.b = None
        self.obj_recent = []
    def ADMM_fit(self):
        m, n, l, totalnum = self.Xu.shape
        X = self.Xu.reshape(-1, totalnum).T
        Xl = self.Xl.reshape(-1, 2 * self.labelednum).T
        L = self.L.flatten()

        XI = np.kron(X, np.ones((totalnum, 1)))
        XJ = np.tile(X, (totalnum, 1))
        XminusX = (XI - XJ) * np.kron(L, np.ones((1, m * n * l)))

        c = 1 / (1 + self.rho + self.lambda_ * np.sum(XminusX * XminusX, axis=1))
        sqrtc = c**2
        coeff1 = (self.rho + 1) * sqrtc - 2 * c
        coeff2 = self.lambda_ * sqrtc

        H = (coeff1 * (Xl @ Xl.T * (self.y[:, np.newaxis] @ self.y[np.newaxis, :]))) + \
            (coeff2 * ((Xl @ XminusX.T) @ (Xl @ XminusX.T) * (self.y[:, np.newaxis] @ self.y[np.newaxis, :])))
        
        s_km1 = np.zeros(m * n * l)
        s_hatk = s_km1.copy()
        lambda_km1 = np.ones(m * n * l)
        lambda_hatk = lambda_km1.copy()
        t_k = 1
        c_km1 = 0

        recent_number = self.max_iter
        recent_idx = 0
        self.obj_recent = np.zeros(recent_number)

        for k in range(self.max_iter):
            h = (1 + coeff1 * (Xl @ (lambda_hatk + self.rho * s_hatk) * self.y) + 
                coeff2 * ((XminusX @ Xl.T) @ (XminusX @ (lambda_hatk + self.rho * s_hatk)) * self.y))

            LB = np.zeros(2 * self.labelednum)
            UB = (1 / (2 * self.labelednum)) * np.ones(2 * self.labelednum)
            Aeq = np.tile(self.y, (2 * self.labelednum, 1))
            beq = np.zeros(2 * self.labelednum)

            result = minimize(lambda alpha: -alpha @ h, np.zeros(2 * self.labelednum),
                              bounds=list(zip(LB, UB)), constraints={'type': 'eq', 'fun': lambda alpha: Aeq @ alpha - beq})
            alpha = result.x

            w_k = c * (lambda_hatk + self.rho * s_hatk + Xl.T @ (alpha * self.y))
            sel = (alpha > 0) & (alpha < (1 / (2 * self.labelednum)))
            self.b = sel @ (self.y - Xl @ w_k) / np.sum(sel)

            W_k = w_k.reshape(m, n, l)
            Lambda_k = lambda_hatk.reshape(m, n, l)

            S, nuc_sumvalue = prox_tnn(self.rho * W_k - Lambda_k, self.tau)
            s_k = (S / self.rho).flatten()

            lambda_k = lambda_hatk - self.rho * (w_k - s_k)

            c_k = np.dot(lambda_k - lambda_hatk, lambda_k - lambda_hatk) / self.rho + \
                  self.rho * np.dot(s_k - s_hatk, s_k - s_hatk)

            self.W = w_k
            self.W_iter.append(w_k)

            if c_k < self.eta * c_km1:
                t_kp1 = 0.5 * (1 + np.sqrt(1 + 4 * t_k * t_k))
                s_hatkp1 = s_k + (t_k - 1) / t_kp1 * (s_k - s_km1)
                lambda_hatkp1 = lambda_k + (t_k - 1) / t_kp1 * (lambda_k - lambda_km1)
                restart = False
            else:
                t_kp1 = 1
                s_hatkp1 = s_km1
                lambda_hatkp1 = lambda_km1
                c_k = c_km1 / self.eta
                restart = True

            s_hatk = s_hatkp1
            lambda_hatk = lambda_hatkp1
            c_km1 = c_k
            s_km1 = s_k
            lambda_km1 = lambda_k
            t_k = t_kp1

            obj_k = obj_value(w_k, self.b, Xl, XminusX, self.y, nuc_sumvalue, self.tau, self.lambda_)
            recent_idx += 1
            self.obj_recent[recent_idx] = obj_k
            if recent_idx == recent_number:
                recent_idx = 0
            
            if k % 1000 == 0:
                rk = np.sum(svd(w_k.reshape(m, n, l)) > 1e-6)
                print(f'k={k}, obj={obj_k:.6f}, restart={restart}, rank={rk}')
            
            if (abs(obj_k - np.mean(self.obj_recent)) / abs(np.mean(self.obj_recent)) < self.eps and k > recent_number):
                break
        print(f'stop_iter {k}')



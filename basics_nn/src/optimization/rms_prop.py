import numpy as np

class RMSprop:
    def __init__(self, lr=0.01, rho=0.9, eps=1e-8):
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.v = None
        
    def update(self, params, grads):
        if self.v is None:
            self.v = np.zeros_like(params)

        self.v = self.rho * self.v + (1 - self.rho) * (grads ** 2)
        params -= self.lr * grads / (np.sqrt(self.v) + self.eps)
        return params
    

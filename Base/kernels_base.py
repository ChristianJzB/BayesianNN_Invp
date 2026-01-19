import torch
import torch.nn as nn

# import numpy as np
# from scipy.optimize import minimize
# from Base.utilities import RootFinder,compute_seq_pairs

class KernelFunction(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def euclidean_distance(self, X, Y):
        return torch.cdist(X, Y)
    
    def d_euclidean_distance(self, X, Y):
        """
        X: (n, d)
        Y: (m, d)
        Returns:
        dr_dX: (n, m, d) - ∂r_{ij}/∂X_i
        dr_dY: (n, m, d) - ∂r_{ij}/∂Y_j
        """
        delta = X[:, None, :] - Y[None, :, :]   # shape: (n, m, d)

        dists = torch.linalg.norm(delta, dim=2)  # shape: (n, m)

        zero_mask = (dists == 0)
        dists_safe = torch.where(zero_mask, torch.ones_like(dists), dists)  # to avoid divide by zero

        dr_dX = delta / dists_safe[..., None]   # shape: (n, m, d)
        dr_dX = torch.where(zero_mask[..., None], torch.zeros_like(dr_dX), dr_dX)

        dr_dY = -dr_dX
        return dr_dX, dr_dY,delta
    
    
    def kernel(self,derivative_order=0):
        raise NotImplementedError("covariance must be implemented in a subclass.")
    
    def covariance(self, X,Y, derivative_order=0):
        d = self.euclidean_distance(X, Y)
        return self.kernel(d, derivative_order=derivative_order)
        

class MaternKernel52(KernelFunction):
    def __init__(self, sigma=1.0, l=1.0, device='cpu'):
        super().__init__(device)
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(sigma, dtype=torch.float64, device=device)))
        self.log_l = nn.Parameter(torch.log(torch.tensor(l, dtype=torch.float64, device=device)))
        self.sqrt5 = torch.sqrt(torch.tensor(5.0, dtype=torch.float64, device=device))
    @property
    def sigma(self):
        return torch.exp(self.log_sigma)

    @property
    def l(self):
        return torch.exp(self.log_l)

    def matern52_kernel(self, r):
        term1 = 1 + self.sqrt5 * r / self.l + 5 * r**2 / (3 * self.l**2)
        return self.sigma**2 * term1 * torch.exp(-self.sqrt5 * r / self.l)
    
    def d_matern52_kernel(self, r):
        term1 = -(5 / 3) * (r / self.l**2) - (self.sqrt5 * 5 * r**2) / (3 * self.l**3)
        return self.sigma**2 * term1 * torch.exp(-self.sqrt5 * r / self.l)

    def dd_matern52_kernel(self, r):
        term1 = -5 / (3 * self.l**2) - (5 / 3) * (r * self.sqrt5 / self.l**3) + (25 * r**2) / (3 * self.l**4)
        return self.sigma**2 * term1 * torch.exp(-self.sqrt5 * r / self.l)
    
    def ddd_matern52_kernel(self, r):
        #term1 = 25 / (3 * self.l**4) + (75 * r)/ (3 * self.l**4) -  (25 * self.sqrt5 * r**2)/ (3 * self.l**5)
        term1 = (75 * r)/ (3 * self.l**4) -  (25 * self.sqrt5 * r**2)/ (3 * self.l**5)
        return self.sigma**2 * term1 * torch.exp(-self.sqrt5 * r / self.l)
    
    def dddd_matern52_kernel(self, r):
        #term1 = 50 / (3 * self.l**4) - (25 * self.sqrt5)/ (3 * self.l**5) + (r**2 * 5**3) / (3*self.l**6)
        term1 = 75 / (3 * self.l**4) - (125 * self.sqrt5*r)/ (3 * self.l**5) + (r**2 * 125) / (3*self.l**6)
        return self.sigma**2 * term1 * torch.exp(-self.sqrt5 * r / self.l)
    
    def d_matern_rrr(self,r):
        term1 = -(5 / 3) * (1/ r*self.l)**2 - (self.sqrt5 * 5 ) / (3 *r* self.l**3)
        zero_mask = r == 0
        term1 = torch.where(zero_mask, torch.zeros_like(term1), term1)
        return self.sigma**2 * term1 * torch.exp(-self.sqrt5 * r / self.l)

    def dd_matern_rr(self,r):
        # Compute all terms with masking for r == 0
        common_term = (25) / (3 * self.l**4)
        mask = r == 0
        # Only apply division where r != 0
        term1 = -5 / (3 * (r*self.l)**2) - (5 / 3) * (self.sqrt5 / (r*self.l**3))
        term1 = torch.where(mask, torch.zeros_like(term1), term1)

        full_term = term1 + common_term
        return self.sigma**2 * full_term * torch.exp(-self.sqrt5 * r / self.l)

    def ddd_matern_r(self,r):
        term1 = 75/ (3 * self.l**4) -  (25 * self.sqrt5 * r)/ (3 * self.l**5)
        return self.sigma**2 * term1 * torch.exp(-self.sqrt5 * r / self.l)
    
    def d_sdd_matern(self,r):
        term1 = -(5 / 3) * (1/self.l)**2 - (5 / 3) * (self.sqrt5*r / self.l**3)
        term2 = 25 / (3 * self.l**4)
        return self.sigma**2 * term1 * torch.exp(-self.sqrt5 * r / self.l), self.sigma**2 * term2 * torch.exp(-self.sqrt5 * r / self.l)
    
    def d_dd_matern(self, r):
        term1 = -25 / (3 * self.l**4)
        return self.sigma**2 * term1 * torch.exp(-self.sqrt5 * r / self.l)
    
    def d_ssdd_matern(self,r):
        term1 = -(10 / 3) * (1/self.l)**2 - (10 / 3) * (self.sqrt5*r / self.l**3) + (25*r**2) / (3 * self.l**4)
        return self.sigma**2 * term1 * torch.exp(-self.sqrt5 * r / self.l)

    def kernel(self, r, derivative_order=0):
        if derivative_order == 0:
            return self.matern52_kernel(r)
        elif derivative_order == 1:
            return self.d_matern52_kernel(r)
        elif derivative_order == 2:
            return self.dd_matern52_kernel(r)
        elif derivative_order == 3:
            return self.ddd_matern52_kernel(r)
        elif derivative_order == 4:
            return self.dddd_matern52_kernel(r)
        elif derivative_order == -1:
            return self.d_matern_rrr(r)
        elif derivative_order == -2:
            return self.dd_matern_rr(r)
        elif derivative_order == -3:
            return self.ddd_matern_r(r)
        elif derivative_order == -4:
            return self.d_sdd_matern(r)
        elif derivative_order == -5:
            return self.d_dd_matern(r)
        elif derivative_order == -6:
            return self.d_ssdd_matern(r)
        else:
            raise ValueError(f"Unsupported derivative order: {derivative_order}")
        


class MaternKernel32(KernelFunction):
    def __init__(self, sigma=1.0, l=1.0, device='cpu'):
        super().__init__(device)
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(sigma, dtype=torch.float64, device=device)))
        self.log_l = nn.Parameter(torch.log(torch.tensor(l, dtype=torch.float64, device=device)))
        self.sqrt3 = torch.sqrt(torch.tensor(3.0, dtype=torch.float64, device=device))

    @property
    def sigma(self):
        return torch.exp(self.log_sigma)

    @property
    def l(self):
        return torch.exp(self.log_l)
    
    def matern32_kernel(self, r):
        term1 = 1 + self.sqrt3 * r / self.l
        return self.sigma**2 * term1 * torch.exp(-self.sqrt3 * r / self.l)
    

    def kernel(self, r, derivative_order=0):
        if derivative_order == 0:
            return self.matern32_kernel(r)
        else:
            raise ValueError(f"Unsupported derivative order: {derivative_order}")


class SquaredExponential(KernelFunction):
    def __init__(self, sigma=1.0, l=1.0, device='cpu'):
        super().__init__(device)
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(sigma, dtype=torch.float64, device=device)))
        self.log_l = nn.Parameter(torch.log(torch.tensor(l, dtype=torch.float64, device=device)))

    @property
    def sigma(self):
        return torch.exp(self.log_sigma)

    @property
    def l(self):
        return torch.exp(self.log_l)
    
    def squared_exponential_cov(self, r):
        return (self.sigma**2) * torch.exp(-0.5 * (r / self.l) ** 2)

    def d_squared_exponential_cov(self, r):
        return -((self.sigma / self.l) ** 2) * r * torch.exp(-0.5 * (r / self.l) ** 2)

    def dd_squared_exponential_cov(self, r):
            return (-1/self.l**2 + r**2/self.l**4 )*self.squared_exponential_cov(r)
    
    def ddd_squared_exponential_cov(self, r):
            return (3*r / self.l**4 - r**3 / self.l**6)*self.squared_exponential_cov(r)
    
    def dddd_squared_exponential_cov(self, r):
            return (3 / self.l**4 - 6*r**2 /self.l**6 + r**4 / self.l**8)*self.squared_exponential_cov(r)

    def kernel(self, r, derivative_order=0):
        if derivative_order == 0:
            return self.squared_exponential_cov(r)
        elif derivative_order == 1:
            return self.d_squared_exponential_cov(r)
        elif derivative_order == 2:
            return self.dd_squared_exponential_cov(r)
        elif derivative_order == 3:
            return self.ddd_squared_exponential_cov(r)
        elif derivative_order == 4:
            return self.dddd_squared_exponential_cov(r)
        else:
            raise ValueError(f"Unsupported derivative order: {derivative_order}")

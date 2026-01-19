import torch
from Base.kernels_base import MaternKernel32, MaternKernel52, SquaredExponential


class GPBase:
    """Base For GPS"""
    def __init__(self, reg_matrix=1e-6,device = "cpu"):
        self.device = device
        self.reg_matrix = reg_matrix
    
    def train_gp(self):
        """Trains the GP"""
        raise NotImplementedError("train_gp must be implemented in a subclass.")
    
    def gp_kernel(self):
        """Trains the GP"""
        raise NotImplementedError("gp_kernel must be implemented in a subclass.")
    

    def g_training(self):
        """Trains the GP"""
        raise NotImplementedError("gp_kernel must be implemented in a subclass.")
    
    def kernel_inverse(self, L, Y):
        return  torch.cholesky_solve(Y, L)  

    def marginal_likelihood(self):
        """Trains the GP"""
        raise NotImplementedError("gp_kernel must be implemented in a subclass.")

    def optimize_mll(self,lr=1e-2):
    # Start from log of current parameters
        params = list(self.kernels.parameters())  # collects all kernels' parameters

        optimizer = torch.optim.LBFGS(params, lr=lr, max_iter=100, line_search_fn='strong_wolfe')

        def closure():
            optimizer.zero_grad()

            nll = -self.marginal_likelihood()
            nll.backward(retain_graph=True)
            return nll

        optimizer.step(closure)

        print("Optimized kernel parameters:")
        for k in self.kernels:
            for name, p in k.named_parameters():
                raw_value = torch.exp(p).item()  # exponentiate
                print(f"{name}: log={p.item():.6f}, exp={raw_value:.6f}")

                p.detach_()

        self.train_gp()

    def marginal_val(self, theta_test):
        """GP"""
        raise NotImplementedError("marginal_val must be implemented in a subclass.")
    
    def prediction(self, theta_test, var=True):
        """Prediction of the GP"""
        raise NotImplementedError("prediction must be implemented in a subclass.")



class VanillaGP(GPBase):
    """Vanilla GPS"""

    def __init__(self, data_training, reg_matrix=1e-6,sigma_l_parameters=(1, 1), kernel = "Matern52",device = "cpu"):
        super().__init__(reg_matrix,device)

        # Training data
        self.parameters_data = torch.tensor(data_training["parameters_data"], dtype=torch.float64,device=self.device)
        self.solutions_data = torch.tensor(data_training["solutions_data"], dtype=torch.float64,device=self.device)

        # Kernel factory
        kernel_dict = {
            "Matern52": MaternKernel52,
            "Matern32": MaternKernel32,
            "SquaredExponential": SquaredExponential
        }

        kernel_class = kernel_dict.get(kernel, MaternKernel52)
        self.kernel_parameter = kernel_class(*sigma_l_parameters, device=self.device)

    def train_gp(self):
        self.g_trained = self.g_training()
        self.cov_matrix, self.L= self.gp_kernel(self.parameters_data, self.parameters_data)
        self.invk_g = self.kernel_inverse(self.L, self.g_trained)

    def gp_kernel(self, theta1, theta2):
        cov = self.kernel_parameter.covariance(theta1, theta2)
        cov =  cov + self.reg_matrix * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
        L = torch.linalg.cholesky(cov)
        return cov,L
    
    def g_training(self):
        return self.solutions_data.view(-1, 1).detach()

    def optimize_gp(self,lr=1e-2):
        self.kernels = torch.nn.ModuleList([self.kernel_parameter])
        self.optimize_mll(lr=lr)

    def marginal_likelihood(self):
        # Recompute covariance fresh each time
        cov, L = self.gp_kernel(self.parameters_data, self.parameters_data)

        y = self.g_trained

        alpha = self.kernel_inverse(L, y)
        logdet_K = 2.0 * torch.sum(torch.log(torch.diag(L)))
        n = y.shape[0]
        two_pi = torch.tensor(2.0 * torch.pi, dtype=y.dtype, device=y.device)

        return -0.5 * ((y.T @ alpha).squeeze() + logdet_K + n * torch.log(two_pi))

    def marginal_val(self, theta_test):
        with torch.inference_mode():
            kernel_param = self.kernel_parameter.covariance(theta_test, self.parameters_data)
            return kernel_param
    
    def prediction(self, theta_test, var=True):
        with torch.inference_mode():
            matrix_test = self.marginal_val(theta_test)
            marginal_mean = matrix_test @ self.invk_g

            if var:
                kernel_param = self.kernel_parameter.covariance(theta_test, theta_test)

                kinv_y = self.kernel_inverse(self.L, matrix_test.T)
                cov =kernel_param - matrix_test @ kinv_y
                return marginal_mean, cov
            return marginal_mean


class MultioutputGP(GPBase):
    def __init__(self, data_training, reg_matrix=1e-6,sigma_l_parameters=(1, 1), kernel = "Matern52",device = "cpu"):
        super().__init__(reg_matrix,device)        
        
        # Training data
        self.parameters_data = torch.tensor(data_training["parameters_data"], dtype=torch.float64,device=self.device)
        self.solutions_data = torch.tensor(data_training["solutions_data"], dtype=torch.float64,device=self.device)
        self.spatial_dim = self.solutions_data.shape[-1]

        self.GPs = [VanillaGP({"parameters_data": self.parameters_data, "solutions_data": self.solutions_data[:, i].reshape(-1, 1)},
                                reg_matrix=reg_matrix,
                                sigma_l_parameters=sigma_l_parameters,
                                kernel = kernel,
                                device=self.device) for i in range(self.spatial_dim)]
        
    def train_gp(self):
        [gp_model.train_gp() for gp_model in self.GPs]

    def optimize_gp(self):
        [gp_model.optimize_gp() for gp_model in self.GPs]

    def prediction(self, theta_test, var=True):
        
        if var:
            predictions = [gp_model.prediction(theta_test,var=True) for gp_model in self.GPs]
            means = [p[0] for p in predictions]       # list of vectors
            covariances = [p[1] for p in predictions] # list of matrices

            return torch.cat(means, dim=1), torch.stack([torch.diag(cov) for cov in covariances], dim=1) 
        else:
            predictions = [gp_model.prediction(theta_test,var=False) for gp_model in self.GPs]
            return  torch.cat(predictions, dim=1)
        

class PIGPBase(GPBase):
    def __init__(self, data_training, reg_matrix=1e-6,kernel_spatial = "Matern52",kernel_parameter = "SquaredExponential",
                 sigma_l_parameters=(1, 1), sigma_l_spatial=(1, 1),device = "cpu"):
        super().__init__(reg_matrix,device)        

        self.device = device
        # Training data
        self.parameters_data = torch.tensor(data_training["parameters_data"], dtype=torch.float64,device=self.device)
        self.solutions_data = torch.tensor(data_training["solutions_data"], dtype=torch.float64,device=self.device)
        self.x_sol_data = torch.tensor(data_training["x_solutions_data"], dtype=torch.float64,device=self.device)

        self.x_bc = torch.tensor(data_training["x_bc"], dtype=torch.float64,device=self.device)
        self.y_bc = torch.tensor(data_training["y_bc"], dtype=torch.float64,device=self.device)

        self.source_func_x = torch.tensor(data_training["source_func_x"], dtype=torch.float64,device=self.device)
        self.source_func_f_x = torch.tensor(data_training["source_func_f_x"], dtype=torch.float64,device=self.device)

        self.n_parameter_obs = self.parameters_data.shape[0]
        self.parameter_dim = self.parameters_data.shape[-1]

        # Kernel factory
        kernel_dict = {
            "Matern52": MaternKernel52,
            "Matern32": MaternKernel32,
            "SquaredExponential": SquaredExponential}

        kernel_class_spatial = kernel_dict.get(kernel_spatial, MaternKernel52)
        self.kernel_spatial = kernel_class_spatial(*sigma_l_spatial, device=self.device)


        kernel_class_parameter = kernel_dict.get(kernel_parameter, MaternKernel52)
        self.kernel_parameter = kernel_class_parameter(*sigma_l_parameters, device=self.device)


    def train_gp(self):
          self.g_trained = self.g_training()
          self.cov_matrix, self.L,self.kuu, self.kug,self.kuf = self.informed_kernel(self.parameters_data, self.parameters_data)
          self.invk_g = self.kernel_inverse(self.L, self.g_trained)

    def optimize_gp(self,lr=1e-2):
        self.kernels = torch.nn.ModuleList([self.kernel_parameter,self.kernel_spatial])
        self.optimize_mll(lr=lr)


    def g_training(self):
        """Define the likelihood function for the coarse model. Must be overridden."""
        raise NotImplementedError("g_training must be implemented in a subclass.")
     
    def exp_kl_eval(self, theta, x):
        """Define the likelihood function for the coarse model. Must be overridden."""
        raise NotImplementedError("exp_kl_eval must be implemented in a subclass.")
     
    def grad_kl_eval(self, theta, x):
        """Define the likelihood function for the coarse model. Must be overridden."""
        raise NotImplementedError("grad_kl_eval must be implemented in a subclass.")
    
    def kernel_uf(self, theta, x):
        """Define the likelihood function for the coarse model. Must be overridden."""
        raise NotImplementedError("kernel_uf must be implemented in a subclass.")
    
    def kernel_ff(self, theta1, theta2):
        """Define the likelihood function for the coarse model. Must be overridden."""
        raise NotImplementedError("kernel_ff must be implemented in a subclass.")
    
    
    def block_matrix_builder_ff(self, th1, th2):
        nth1, nth2 = th1.shape[0], th2.shape[0]
        blocks = []

        for i in range(nth1):
            row_blocks = []
            for j in range(nth2):
                theta1 = th1[i, :]
                theta2 = th2[j, :]
                cov_scalar = self.kernel_parameter.covariance(theta1.view(1, -1), theta2.view(1, -1))
                pde_matrix = self.kernel_ff(theta1.unsqueeze(0), theta2.unsqueeze(0))
                row_blocks.append(cov_scalar * pde_matrix)
            blocks.append(torch.cat(row_blocks, dim=1))
        return torch.cat(blocks, dim=0)
    
    def block_matrix_builder_uf(self, th1, th2, pde_matrices ,x):
        cov_matrix = self.kernel_parameter.covariance(th1, th2)  # [nth1, nth2]

        # Broadcast cov scalars over PDEs
        # [nth1, nth2, 1, 1] * [1, nth2, N, S] → [nth1, nth2, N, S]
        blocks = cov_matrix.unsqueeze(-1).unsqueeze(-1) * pde_matrices.unsqueeze(0)

        # Stack the final block matrix
        result = torch.cat(
            [blocks[i].transpose(0, 1).reshape(x.shape[0], -1) for i in range(th1.shape[0])],
            dim=0
        )  # shape: [nth1 * N, nth2 * S]
        return result
    
    def informed_kernel(self, theta1, theta2):
        kernel_param = self.kernel_parameter.covariance(theta1, theta2)

        kuu = self.kernel_spatial.covariance(self.x_sol_data, self.x_sol_data)
        Kuu = torch.kron(kernel_param, kuu)

        kug = self.kernel_spatial.covariance(self.x_sol_data, self.x_bc)
        Kug = torch.kron(kernel_param, kug)

        kuf = self.kernel_uf(self.parameters_data,self.x_sol_data)
        Kuf = self.block_matrix_builder_uf(theta1, theta2, pde_matrices=kuf, x=self.x_sol_data)

        kgg = self.kernel_spatial.covariance(self.x_bc, self.x_bc)
        Kgg = torch.kron(kernel_param, kgg)

        kgf = self.kernel_uf(self.parameters_data,self.x_bc)
        Kgf = self.block_matrix_builder_uf(theta1, theta2,pde_matrices=kgf,x=self.x_bc)

        Kff = self.block_matrix_builder_ff(theta1, theta2)

        top = torch.cat([Kuu, Kug, Kuf], dim=1)
        middle = torch.cat([Kug.T, Kgg, Kgf], dim=1)
        bottom = torch.cat([Kuf.T, Kgf.T, Kff], dim=1)
        cov = torch.cat([top, middle, bottom], dim=0)

        cov = cov + self.reg_matrix * torch.eye(cov.shape[0], dtype=torch.float64, device=self.device)
        L = torch.linalg.cholesky(cov)
        return cov,L,kuu,kug,kuf

    def marginal_likelihood(self):
        # Recompute the covariance matrix
        _,L, _, _,_ = self.informed_kernel(self.parameters_data, self.parameters_data)
        
        # Solve K^{-1}y
        y = self.g_trained
        alpha =self.kernel_inverse(L,y)

        # Log determinant
        logdet_K = 2.0 * torch.sum(torch.log(torch.diag(L)))
        two_pi = torch.tensor(2.0 * torch.pi, dtype=y.dtype, device=y.device)
        return -0.5 * ((y.T @ alpha).squeeze() + logdet_K + y.shape[0] * torch.log(two_pi))
    

    def marginal_val(self, theta_test, x_test=None):
        with torch.inference_mode():
            kernel_param = self.kernel_parameter.covariance(theta_test, self.parameters_data)

            if x_test is None:
                x_test = self.x_sol_data
                Kuf = self.block_matrix_builder_uf(theta_test, self.parameters_data,pde_matrices=self.kuf,x=x_test)
                kuu = self.kuu
                kug = self.kug
            else: 
                kuf = self.kernel_uf(self.parameters_data,x_test)
                Kuf = self.block_matrix_builder_uf(theta_test, self.parameters_data,pde_matrices=kuf,x=x_test)

                kuu = self.kernel_spatial.covariance(x_test, self.x_sol_data)
                
                kug = self.kernel_spatial.covariance(x_test, self.x_bc)

            Kuu = torch.kron(kernel_param, kuu)

            Kug = torch.kron(kernel_param, kug)
            
            return torch.cat([Kuu, Kug, Kuf], dim=1)

    def prediction(self, theta_test, x_test=None, var=True):
        with torch.inference_mode():
            matrix_test = self.marginal_val(theta_test, x_test)
            marginal_mean = matrix_test @ self.invk_g

            if var:
                kernel_spatial = self.kuu if x_test is None else  self.kernel_spatial.covariance(x_test, x_test)
                kernel_param = self.kernel_parameter.covariance(theta_test, theta_test)
                kinv_y = self.kernel_inverse(self.L, matrix_test.T)
                cov = torch.kron(kernel_param, kernel_spatial) - matrix_test @ kinv_y
                return marginal_mean, cov
            return marginal_mean
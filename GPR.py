import numpy as np

class GPR: 
    def __init__(self,cov_function_name, x_train, y_train, hyper_params, sigma_n, warp_params = None, nu=None):
        self.cov_function_name = cov_function_name
        self.x_train = x_train
        self.y_train = y_train
        self.hyper_params = hyper_params #params[0] is signal variance, params[1] is lengthscale
        self.nu = nu
        self.K = self.cov_func(self.x_train,self.x_train)
        self.sigma_n = sigma_n
        self.K_inv = np.linalg.pinv(self.K+np.eye(self.K.shape[0])*(self.sigma_n**2))
        if warp_params is None:
            self.f = y_train
            self.df_dy = 1
        else:
            self.gamma = warp_params
            self.f = 0
            self.df_dy = 0
            self.a0 = self.gamma[0]
            self.b0 = self.gamma[1]
            self.c0 = self.gamma[2]
            for i in range(len(self.gamma[0])):
                self.f += self.a0[i] * np.tanh(self.b0[i] * (y_train + self.c0[i])) 
                self.df_dy += self.a0[i] * self.b0[i] * (1 - np.tanh(self.b0[i] * (y_train + self.c0[i])) ** 2)

    def predict(self,x_star):
        k_star = self.cov_func(self.x_train,x_star)
        f_mean = np.dot(np.dot(np.transpose(k_star),self.K_inv),self.f)
        f_std = self.cov_func(x_star, x_star) - np.dot(np.dot(np.transpose(k_star),self.K_inv),k_star)
        f_std = np.sqrt(np.diag(f_std))[:,np.newaxis]
        return [f_mean,f_std]


    def predict_original(self,x_star):
        k_star = self.cov_func(self.x_train,x_star)
        y_mean = np.dot(np.dot(np.transpose(k_star),self.K_inv),self.y_train)
        y_std = self.cov_func(x_star, x_star) - np.dot(np.dot(np.transpose(k_star),self.K_inv),k_star)
        y_std = np.sqrt(np.diag(y_std))[:,np.newaxis]
        return [y_mean,y_std]

    def cov_func(self,x_1,x_2):
        
        x_1_sq = np.sum(np.square(x_1),1)
        x_2_sq = np.sum(np.square(x_2),1)
        d = -2.*np.dot(x_1, x_2.T) + (x_1_sq[:,None] + x_2_sq[None,:])
        d = np.sqrt(np.clip(d, 0, np.inf))

        if self.cov_function_name == "Squared Exponential":
            K = self.hyper_params[0]* np.exp(-0.5 * (d/self.hyper_params[1])**2)

        if self.cov_function_name == "Matern":

            if self.nu == 5/2:
                K = self.hyper_params[0] *(1 + np.sqrt(5) * d / self.hyper_params[1] + 5 * d ** 2 / (3 * self.hyper_params[1] ** 2)) \
                        * np.exp(-np.sqrt(5) * d / self.hyper_params[1])

            elif self.nu == 1/2:
                K = self.hyper_params[0] *np.exp(-0.5*(d/self.hyper_params[1])**2)
    
            else:
                K = self.hyper_params[0] *np.exp(-0.5*(d/self.hyper_params[1])**2)

                print('invalid nu')
                
        return K

    def set_hyper_params(self, hyper_params, sigma_n, warp_params = None):
        self.hyper_params = hyper_params
        self.K = self.cov_func(self.x_train,self.x_train)
        self.sigma_n = sigma_n
        self.K_inv = np.linalg.pinv(self.K+np.eye(self.K.shape[0])*(self.sigma_n**2))
        if warp_params is None:
            self.df_dy = 1
        else:
            self.gamma = warp_params    
            self.a = self.gamma[0]
            self.b = self.gamma[1]
            self.c = self.gamma[2]
            self.f = 0
            self.df_dy = 0
            for i in range(len(self.gamma[0])):
                self.f += self.a[i] * np.tanh(self.b[i] * (self.y_train + self.c[i])) 
                self.df_dy += self.a[i] * self.b[i] * (1 - np.tanh(self.b[i] * (self.y_train + self.c[i])) ** 2)
                
    def log_marginal_likelihood(self):
        lml1 = -.5*np.dot(np.dot(np.transpose(self.f),self.K_inv),self.f)
        lml2 = -.5*np.log(np.linalg.det(self.K+np.eye(self.K.shape[0])*(self.sigma_n**2)))
        lml3 = -.5*self.x_train.shape[0]*np.log(2*np.pi)
        lml4 = np.sum(np.log(self.df_dy))
        return lml1+lml2+lml3+lml4

    def log_marginal_likelihood_original(self):
        lml1 = -.5*np.dot(np.dot(np.transpose(self.y_train),self.K_inv),self.y_train)
        lml2 = -.5*np.log(np.linalg.det(self.K+np.eye(self.K.shape[0])*(self.sigma_n**2)))
        lml3 = -.5*self.x_train.shape[0]*np.log(2*np.pi)
        return lml1+lml2+lml3

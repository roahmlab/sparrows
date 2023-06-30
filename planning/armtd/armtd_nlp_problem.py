import torch
import numpy as np
import time

class OfflineArmtdFoConstraints:
    def __init__(self, dimension = 3, dtype = torch.float):
        self.dimension = dimension
        self.dtype = dtype
        self.np_dtype = torch.empty(0,dtype=dtype).numpy().dtype
    
    def set_params(self, FO_link_zono, A, b, g_ka, n_obs_in_FO, n_joints):
        self.FO_link_zono = FO_link_zono
        self.A = A
        self.b = b
        self.g_ka = g_ka
        self.n_obs_in_FO = n_obs_in_FO
        self.n_links = len(FO_link_zono)
        self.n_timesteps = FO_link_zono[0].batch_shape[0]
        self.n_obs_cons = self.n_timesteps * n_obs_in_FO
        self.M = self.n_links * self.n_obs_cons
        self.n_joints = n_joints

    def __call__(self, x, Cons_out=None, Jac_out=None):
        x = torch.as_tensor(x, dtype=self.dtype)
        if Cons_out is None:
            Cons_out = np.empty(self.M, dtype=self.np_dtype)
        if Jac_out is None:
            Jac_out = np.empty((self.M, self.n_joints), dtype=self.np_dtype)

        for j in range(self.n_links):
            # slice the center and get the gradient for it
            c_k = self.FO_link_zono[j].center_slice_all_dep(x)
            grad_c_k = self.FO_link_zono[j].grad_center_slice_all_dep(x)

            # use those to compute the halfspace constraints
            h_obs = (self.A[j]@c_k.unsqueeze(-1)).squeeze(-1) - self.b[j]
            cons_obs, ind = torch.max(h_obs.nan_to_num(-torch.inf),-1)
            ind = ind.reshape(self.n_obs_in_FO,self.n_timesteps,1,1).expand(-1,-1,-1,self.dimension)
            A_max = self.A[j].gather(-2, ind)
            grad_obs = (A_max@grad_c_k).reshape(self.n_obs_cons, self.n_joints)

            Cons_out[j*self.n_obs_cons:(j+1)*self.n_obs_cons] = -cons_obs.reshape(self.n_obs_cons).numpy()
            Jac_out[j*self.n_obs_cons:(j+1)*self.n_obs_cons] = -grad_obs.numpy()
        
        return Cons_out, Jac_out

"""
Load trigonometry version of precomtuted joint reacheable set (precomputed by CORA)
Author: Yongseok Kwon
Reference: Holmes, Patrick, et al. ARMTD
"""
import torch
from zonopy import gen_rotatotope_from_jrs_trig, gen_batch_rotatotope_from_jrs_trig
#from zonopy.transformations.homogeneous import gen_batch_H_from_jrs_trig
from zonopy import zonotope, polyZonotope, batchZonotope
from scipy.io import loadmat
import os


T_fail_safe = 0.5

dirname = os.path.dirname(__file__)
jrs_tensor_path = os.path.join(dirname,'jrs_trig_tensor_saved/')

JRS_KEY = loadmat(jrs_tensor_path+'c_kvi.mat')
g_ka = loadmat(jrs_tensor_path+'d_kai.mat')['d_kai'][0,0]
#JRS_KEY = torch.tensor(JRS_KEY['c_kvi'],dtype=torch.float)

'''
qjrs_path = os.path.join(dirname,'qjrs_mat_saved/')
qjrs_key = loadmat(qjrs_path+'c_kvi.mat')
qjrs_key = torch.tensor(qjrs_key['c_kvi'])
'''
cos_dim = 0 
sin_dim = 1
vel_dim = 2
ka_dim = 3
acc_dim = 3 
kv_dim = 4
time_dim = 5

# TODO VALIDATE
def preload_batch_JRS_trig(dtype=torch.float,device='cpu'):
    jrs_tensor = []
    for c_kv in JRS_KEY['c_kvi'][0]:
        jrs_filename = jrs_tensor_path+'jrs_trig_tensor_mat_'+format(c_kv,'.3f')+'.mat'
        jrs_tensor_load = loadmat(jrs_filename)
        jrs_tensor.append(jrs_tensor_load['JRS_tensor'].tolist()) 
    return torch.tensor(jrs_tensor,dtype=dtype,device=device)


import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from garage.np.policies import StableCartSpringDamperPolicy
from garage.np.algos.mod_cem_ssd import MOD_CEM_SSD
from yumikin.YumiKinematics import YumiKinematics
from rllab.envs.gym_env import GymEnv
from utils import Sample
# from gym.envs.mujoco.yumipeg import GOAL
from gps.agent.ros.agent_ros import AgentROS
from gps.proto.gps_pb2 import TRIAL_ARM, AUXILIARY_ARM, JOINT_SPACE
from agent_hyperparams import agent as agent_params
from agent_hyperparams import reward_params
import copy

base_log_file = '/home/shahbaz/Research/Software/Stablevic27/log/'
exp_name = 'peg_x_2'
log_dir = base_log_file+exp_name
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

init_pos = agent_params['x0'][0][:7]
GOAL_POS = np.array([-2.0425, -0.7721, 2.0033, -0.0046, 1.9164, 1.0043, -1.2424])
# GOAL_POS = init_pos



kin_params_yumi = {}
kin_params_yumi['urdf'] = '/home/shahbaz/Research/Software/yumi_kinematics/yumikin/models/yumi_ABB_left.urdf'
kin_params_yumi['base_link'] = 'world'
kin_params_yumi['end_link'] = 'left_tool0'
# kin_params_yumi['end_link'] = 'left_contact_point'
kin_params_yumi['euler_string'] = 'sxyz'
kin_params_yumi['goal'] = GOAL_POS
yumiKin = YumiKinematics(kin_params_yumi)

x_d_i,_,_ = yumiKin.get_cart_error_frame_terms(init_pos,np.zeros(7))

delat_goal_cart_dist = x_d_i[:3]
delat_goal_rot_dist = x_d_i[3:]
assert(np.any(np.abs(delat_goal_rot_dist)<np.pi/2)) # check if any rotation coordinate is more than pi/2
r_cart_sq = np.square(np.linalg.norm(delat_goal_cart_dist))
r_cart_comp_sq = r_cart_sq*np.ones(3)
r_rot_sq = np.square(np.pi/4) # so that 2-sigma is pi/2
r_rot_comp_sq = r_rot_sq*np.ones(3)
init_mu_cov_diag = np.concatenate((r_cart_comp_sq, r_rot_comp_sq))
init_mu_mu =np.zeros(6)

M_norm = 0.4 # a value between 0 and 1.
# s_trans = 200.
# s_rot = 5.
s_trans = 200.
s_rot = 4.
S0_init = np.diag(np.array([s_trans, s_trans, s_trans, s_rot, s_rot, s_rot]))
M_d_x = yumiKin.get_cart_intertia_d(init_pos)
M_d = np.diag(M_d_x)
D_d = np.sqrt(np.multiply(M_d, np.diag(S0_init)))
print('D_init:', D_d)
d_trans = np.max(D_d[:3])
d_rot = np.max(D_d[3:])

#s,d derivation
T = 100
dt = 0.01
t = T*dt
b = -2*M_d/t
k = np.square(b)/(4.*M_d)


# start with v=20 and expect convergence with v=5000. Starting mean 0.4 has full coverage from 0.1 to 1. at v=20
# this is the base p.d. matrix and will be used to scale 4 3X3 matrices for S/D_trans, S/D_rot.
resume = True

SD_mat_init = {}
SD_mat_init['M_init'] = M_norm
SD_mat_init['D_trans_s'] = d_trans/M_norm
SD_mat_init['D_rot_s'] = d_rot/M_norm
SD_mat_init['S_trans_s'] = s_trans/M_norm
SD_mat_init['S_rot_s'] = s_rot/M_norm
SD_mat_init['v'] = 30.
SD_mat_init['local_scale'] = 4.

T = agent_params['T']
n_samples = 15
n_epochs = 40
entropy_const=1.0e1
v_scalar_init = 20
K=2
best_frac=0.2
init_cov_diag=init_mu_cov_diag
SD_mat_init = SD_mat_init
elite=True
temperature=.1
entropy_step_v=100

algo_params = {}
algo_params['n_samples'] = n_samples
algo_params['n_epochs'] = n_epochs
algo_params['init_mu_mu'] = init_mu_mu
algo_params['init_mu_cov_diag'] = init_mu_cov_diag
algo_params['init_pos'] = init_pos
algo_params['goal_pos'] = GOAL_POS
algo_params['SD_mat_init'] = SD_mat_init
algo_params['K'] = K
algo_params['best_frac'] = best_frac
algo_params['v_scalar_init'] = v_scalar_init
algo_params['elite'] = elite
algo_params['temperature'] = temperature
algo_params['entropy_const'] = entropy_const
algo_params['entropy_step_v'] = entropy_step_v

exp_params = {}
exp_params['kinparams'] = kin_params_yumi
exp_params['agent_params'] = agent_params
exp_params['algo_params'] = algo_params
exp_params['reward_params'] = reward_params

if resume==False:
    policy = StableCartSpringDamperPolicy(GOAL_POS, kin_params_yumi, T, K=K)
    algo = MOD_CEM_SSD(
                        policy=policy,
                        # best_frac=0.05,
                        best_frac=best_frac,
                        n_samples=n_samples,
                        init_cov_diag=init_mu_cov_diag,
                        SD_mat_init = SD_mat_init,
                        v_scalar_init = v_scalar_init,
                        mu_init = init_mu_mu,
                        elite=elite,
                        temperature=temperature,
                        # entropy_const=1e1,
                        entropy_const=entropy_const,
                        entropy_step_v=entropy_step_v,
                        )

    algo.train()
    agent = AgentROS(agent_params)
    data_log = []
    ep_start = 0
else:
    exp_state = pickle.load(open(log_dir+'/'+'data', "rb"))
    exp_params = exp_state['exp_params']
    algo_params = exp_params['algo_params']
    agent_params = exp_params['agent_params']

    GOAL_POS = algo_params['goal_pos']
    K = algo_params['K']
    T = agent_params['T']
    policy = StableCartSpringDamperPolicy(GOAL_POS, kin_params_yumi, T, K=K)

    n_samples = algo_params['n_samples']
    n_epochs = algo_params['n_epochs']
    init_mu_mu = algo_params['init_mu_mu']
    init_mu_cov_diag = algo_params['init_mu_cov_diag']
    SD_mat_init = algo_params['SD_mat_init']
    best_frac = algo_params['best_frac']
    v_scalar_init = algo_params['v_scalar_init']
    elite = algo_params['elite']
    temperature = algo_params['temperature']
    entropy_const = algo_params['entropy_const']
    entropy_step_v = algo_params['entropy_step_v']
    algo = MOD_CEM_SSD(
                        policy=policy,
                        # best_frac=0.05,
                        best_frac=best_frac,
                        n_samples=n_samples,
                        init_cov_diag=init_mu_cov_diag,
                        SD_mat_init = SD_mat_init,
                        v_scalar_init = v_scalar_init,
                        mu_init = init_mu_mu,
                        elite=elite,
                        temperature=temperature,
                        # entropy_const=1e1,
                        entropy_const=entropy_const,
                        entropy_step_v=entropy_step_v,
                        )
    algo.train()
    algo.cur_params = exp_state['algo_cur_params']
    algo.cur_stat = exp_state['algo_cur_stat']
    algo.set_params(algo.cur_params)
    agent = AgentROS(agent_params)
    data_log = exp_state['data']
    ep_start = len(data_log)

for ep in range(ep_start, n_epochs):
    # print('Epoch:',ep)
    exp_state = {}
    epoc_samples = []
    for n in range(n_samples):
        print('Epoch:', ep)
        print('Sample:', n)
        itr = ep*n_samples + n
        path = agent.sample(policy, 0, verbose=True, save=False, noisy=False)
        epoc_samples.append(path)
        algo.train_once(itr, path)
    data_log.append(epoc_samples)
    exp_state['data'] = data_log
    exp_state['algo_cur_stat'] = copy.copy(algo.cur_stat)
    exp_state['algo_cur_params'] = copy.copy(algo.cur_params)
    exp_state['exp_params'] = exp_params
    pickle.dump(exp_state, open(log_dir+'/'+'data', "wb"))


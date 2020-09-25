
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from garage.np.policies import StableCartSpringDamperPolicy
from garage.np.algos.mod_cem_ssd import MOD_CEM_SSD
from yumikin.YumiKinematics import YumiKinematics
from rllab.envs.gym_env import GymEnv
from utils import Sample
from gym.envs.mujoco.yumipeg import GOAL

base_log_file = '/home/shahbaz/Research/Software/Stablevic27/log/'
exp_name = 'test'
log_file = base_log_file+exp_name
if not os.path.exists(log_file):
    os.mkdir(log_file)

GOAL_POS = GOAL
# INIT_POS = np.array([-1.63688, -1.14705, 0.93536, 0.622845, 1.96716, 1.86328, 0.47748])
kin_params_yumi = {}
kin_params_yumi['urdf'] = '/home/shahbaz/Research/Software/yumi_kinematics/yumikin/models/yumi_ABB_left.urdf'
kin_params_yumi['base_link'] = 'world'
# kin_params_yumi['end_link'] = 'gripper_l_base'
kin_params_yumi['end_link'] = 'left_contact_point'
kin_params_yumi['euler_string'] = 'sxyz'
kin_params_yumi['goal'] = GOAL_POS
yumiKin = YumiKinematics(kin_params_yumi)
T = 100
dt = 0.20
n_samples = 15
n_epochs = 40
dJS = 14
dJA = 7
plot = True

env = GymEnv("YumiPeg-v1")
policy = StableCartSpringDamperPolicy(GOAL_POS, kin_params_yumi, T, K=2)
algo = MOD_CEM_SSD(
                    policy=policy,
                    # best_frac=0.05,
                    best_frac=0.2,
                    n_samples=n_samples,
                    init_cov_diag=1.,
                    S_init=2,
                    elite=True,
                    temperature=.1,
                    # entropy_const=1e1,
                    entropy_const=1.2e1,
                    entropy_step_v=100,
                    )
algo.train()

exp_log = []
for ep in range(n_epochs):
    print('Epoch:',ep)
    epoc_samples = []
    for n in range(n_samples):
        # print('Sample:', n)
        itr = ep*n
        path = Sample(env, algo.policy, T, dJS, dJA, plot=(plot and (n==0)))
        epoc_samples.append(path)
        algo.train_once(itr, path)
    exp_log.append(epoc_samples)
pickle.dump(exp_log, open(log_file+'/exp_log', "wb"))




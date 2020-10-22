
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
#from garage.np.policies import StableCartSpringDamperPolicy
#from garage.np.algos.mod_cem_ssd import MOD_CEM_SSD
from yumikin.YumiKinematics import YumiKinematics
from utils import Sample
# from gym.envs.mujoco.yumipeg import GOAL
from gps.agent.ros.agent_ros import AgentROS
from gps.proto.gps_pb2 import TRIAL_ARM, AUXILIARY_ARM, JOINT_SPACE
from agent_hyperparams import agent as agent_params
from agent_hyperparams import reward_params
import copy
from policy import Policy

base_log_file = '/home/shahbaz/Software/Stablevic27/log/'
exp_name = 'peg_x_2'
log_dir = base_log_file+exp_name
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

init_pos = agent_params['x0'][0][:7]
GOAL_POS = np.array([-2.0425, -0.7721, 2.0033, -0.0046, 1.9164, 1.0043, -1.2424])
# GOAL_POS = init_pos



kin_params_yumi = {}
kin_params_yumi['urdf'] = '/home/shahbaz/Software/yumi_kinematics/yumikin/models/yumi_ABB_left.urdf'
kin_params_yumi['base_link'] = 'world'
kin_params_yumi['end_link'] = 'left_tool0'
# kin_params_yumi['end_link'] = 'left_contact_point'
kin_params_yumi['euler_string'] = 'sxyz'
kin_params_yumi['goal'] = GOAL_POS
yumiKin = YumiKinematics(kin_params_yumi)

T = agent_params['T']
n_samples = 1
n_epochs = 1

policy = Policy()

agent = AgentROS(agent_params)
data_log = []
ep_start = 0

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
    data_log.append(epoc_samples)
    exp_state['data'] = data_log
    pickle.dump(exp_state, open(log_dir+'/'+'data', "wb"))


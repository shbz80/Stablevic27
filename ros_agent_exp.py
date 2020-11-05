
import numpy as np
from gps.agent.ros.agent_ros import AgentROS
from agent_hyperparams import agent as agent_params
from agent_hyperparams import reward_params

T = agent_params['T']
n_samples = 1
n_epochs = 1

policy = Policy()

agent = AgentROS(agent_params)
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


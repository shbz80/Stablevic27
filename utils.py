import numpy as np

def Sample(env, policy, T, dS, dA, plot=True):
    s = env.reset()
    assert(s.shape==(dS,))
    S = np.zeros((T,dS))
    A = np.zeros((T,dA))
    A_info = np.zeros((T,6))
    path = {}
    path['agent_infos']={}

    for t in range(T):
        a, a_info = policy.get_action(s)
        next_s, _, _, _ = env.step(a)
        if plot:
            env.render()
        S[t] = s
        A[t] = a
        A_info[t] = a_info['mean']
        s = next_s
    path['observations'] = S
    path['actions'] = A
    path['agent_infos']['mean'] = A_info
    return path
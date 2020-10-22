import numpy as np
class Policy ():
    def __init__(self):
        self.dU = 7
        return

    def reset(self):
        return

    def act(self, x, obs, t, noise=None):
        print('Obs:',obs)
        torque = 0.01*np.ones(7)
        return torque, torque




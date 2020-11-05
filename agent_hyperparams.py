""" Hyperparameters for yumi robot trajectory optimization experiment.
with dither insertion task, tweeked YUMI_GAINS for improved exploration
final cost multiplier 10*2, adjusted yumi gains but a drift observed on j4
due to dithering.
"""
from __future__ import division

from datetime import datetime
import os.path
import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.ros.agent_ros import AgentROS
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION, \
        TRIAL_ARM, AUXILIARY_ARM, JOINT_SPACE
from gps.utility.general_utils import get_ee_points
# from gps.gui.config import generate_experiment_info


# EE_POINTS = np.array([[0.02, -0.025, 0.05], [0.02, -0.025, -0.05],
#                      [0.02, 0.05, 0.0]])
EE_POINTS = np.array([[0., 0.0, 0.], [0., 0., 0.],
                      [0., 0., 0.]])

# SENSOR_DIMS = {
#     JOINT_ANGLES: 7,
#     JOINT_VELOCITIES: 7,
#     END_EFFECTOR_POINTS: 3 * EE_POINTS.shape[0],
#     END_EFFECTOR_POINT_VELOCITIES: 3 * EE_POINTS.shape[0],
#     ACTION: 7,
# }
SENSOR_DIMS = {
    JOINT_ANGLES: 7,
    JOINT_VELOCITIES: 7,
    ACTION: 7,
}

x0s = []
ee_tgts = []
reset_conditions = []

common = {
    'conditions': 1,
}

# TODO(chelsea/zoe) : Move this code to a utility function
# Set up each condition.
for i in range(common['conditions']):
    # ja_x0 = np.array([-1.2711, -1.0145, 1.2650, 0.6863, 2.0623, 1.2777, -0.5737]) exp_peg_1
    ja_x0 = np.array([-1.5942, -0.7852, 1.5535, 0.3930, 1.9778, 1.1883, -1.2780]) #exp_peg_1

    # just before touching the bracket
    ee_pos_tgt = np.array([[ 0.454288,  0.268467,  0.172322]])
    ee_rot_tgt = np.array([[[ -0.00875064,  0.999562,  -0.0282783],
                        [ -0.00722514,  0.0282155,  0.999576],
                        [ 0.999936,  0.00895124,  0.00697507]]])

    x0 = np.zeros(14)
    x0[:7] = ja_x0
    ee_tgt = np.ndarray.flatten(
        get_ee_points(EE_POINTS, ee_pos_tgt, ee_rot_tgt).T
    )
    aux_x0 = np.array([[ 0.0,  -2.268928,  -2.356194,  0.523599,  0.0,  0.698132,  0.0]])

    reset_condition = {
        TRIAL_ARM: {
            'mode': JOINT_SPACE,
            'data': x0[0:7],
        },
        AUXILIARY_ARM: {
            'mode': JOINT_SPACE,
            'data': aux_x0,
        },
    }

    x0s.append(x0)
    ee_tgts.append(ee_tgt)
    reset_conditions.append(reset_condition)

reward_params = {}
reward_params['LIN_SCALE'] = 1
reward_params['ROT_SCALE'] = 1
reward_params['POS_SCALE'] = 1
reward_params['VEL_SCALE'] = 1e-1
reward_params['STATE_SCALE'] = 1
reward_params['ACTION_SCALE'] = 1e-2
reward_params['v'] = 2
reward_params['w'] = 1
reward_params['TERMINAL_STATE_SCALE'] = 20

agent = {
    'type': AgentROS,
    'dt': 0.01,
    'conditions': common['conditions'],
    'T': 500,
    'trial_timeout': 7,
    'reset_timeout': 50,
    'x0': x0s,
    'ee_points_tgt': ee_tgts,
    'reset_conditions': reset_conditions,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES],
    'end_effector_points': EE_POINTS,
    'obs_include': [],
}




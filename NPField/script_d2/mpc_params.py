"""Centralized MPC / motion parameters for D2 pipeline.

This file is the single source of truth for:
- MPC dimensions, bounds, and weights
- solver configuration
- agent and obstacle motion timing/speed
- geometry constants used by plotting and obstacle footprints
"""

# Horizon and map/potential settings
N_HORIZON = 30
TIME_STEPS = 10
POTENTIAL_RESOLUTION = 100
SOLVER_BASE_TF = 25.0
MAP_SCALE = 10.0

# Agent motion settings
V_MAX = 0.9

# Obstacle prediction/motion settings shared by NN, MPC, and environment
OBSTACLE_PRED_DT = 0.8  # seconds between NN obstacle predictions
OBSTACLE_SPEED_MPS = 0.3  # dynamic obstacle speed (m/s)
OBSTACLE_STEP_M = OBSTACLE_SPEED_MPS * OBSTACLE_PRED_DT

# MPC model dimensions
MPC_NX = 5
MPC_NU = 3
MPC_OBSTACLE_COST_DIM = 1

# State bounds [x, y, v, theta, t_point]
STATE_LBX = [-100.0, -100.0, 0.0, -100.0, 0.0]
STATE_UBX = [100.0, 100.0, 1.0, 100.0, 100.0]

# Control bounds [a, w, T]
CTRL_A_MIN = -0.35
CTRL_A_MAX = 0.35
CTRL_W_MIN = -0.8
CTRL_W_MAX = 0.8
CTRL_T_MIN = 0.0
CTRL_T_MAX = 10.0

# Cost weights
W_X = 0.1
W_Y = 0.1
W_V = 0.05
W_THETA = 0.1
W_TIME = 0.01
W_A = 0.01
W_W = 0.005
W_T = 0.01
W_OBST = 300.0

W_X_E = 30.0
W_Y_E = 30.0
W_V_E = 0.001
W_THETA_E = 0.01
W_TIME_E = 0.01

# Solver settings
NLP_MAX_ITER = 80
QP_MAX_ITER = 100
NLP_TOL_STAT = 1e-4
NLP_TOL_EQ = 1e-4
NLP_TOL_INEQ = 1e-4
NLP_TOL_COMP = 1e-4
LM_DAMPING = 3.0
QP_COND_N = 10

# Time allocation robustness for finite horizon
TF_TIME_SLACK = 1.25
TF_TURN_WEIGHT = 1.0
TF_MIN_BUFFER_SEC = 1.0

# Geometry settings used in environment/plot overlays
OBSTACLE_FOOTPRINT_RADIUS = 0.291
OBSTACLE_FOOTPRINT_ANGLE = 1.03

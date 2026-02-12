from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from robot_model_GPT import robot_model
import numpy as np
import l4casadi as l4c
import torch
from mpc_params import (
    CTRL_A_MAX,
    CTRL_A_MIN,
    CTRL_T_MAX,
    CTRL_T_MIN,
    CTRL_W_MAX,
    CTRL_W_MIN,
    LM_DAMPING,
    MPC_NU,
    MPC_NX,
    NLP_MAX_ITER,
    NLP_TOL_COMP,
    NLP_TOL_EQ,
    NLP_TOL_INEQ,
    NLP_TOL_STAT,
    N_HORIZON,
    QP_COND_N,
    QP_MAX_ITER,
    SOLVER_BASE_TF,
    STATE_LBX,
    STATE_UBX,
    W_A,
    W_OBST,
    W_T,
    W_THETA,
    W_THETA_E,
    W_TIME,
    W_TIME_E,
    W_V,
    W_V_E,
    W_W,
    W_X,
    W_X_E,
    W_Y,
    W_Y_E,
)

def create_solver(model_loaded, embedding_values):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_loaded.to(device)
    model, l4c_model = robot_model(model_loaded, embedding_values)

    # acados OCP handle
    ocp = AcadosOcp()
    N = N_HORIZON
    ocp.dims.N = N

    # OCP dimensions
    nx = MPC_NX
    nu = MPC_NU
    ny = nx + nu

    # OCP costs
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.model.cost_y_expr = model.cost_y_expr
    ocp.model.cost_y_expr_e = model.cost_y_expr_e

    ######## set weights for cost function
    W_obst = np.array([W_OBST])
    W_x = np.array([W_X, W_Y, W_V, W_THETA, W_TIME, W_A, W_W, W_T])
    W = np.diag(np.concatenate([W_x, W_obst]))
    W_xe = np.array([W_X_E, W_Y_E, W_V_E, W_THETA_E, W_TIME_E])
    W_e = np.diag(np.concatenate([W_xe, W_obst]))

    ocp.cost.W = W
    ocp.cost.W_e = W_e
    ocp.cost.yref = np.zeros([ny + 1])
    ocp.cost.yref_e = np.zeros([nx + 1])

    ocp.constraints.idxbx = np.array([0, 1, 2, 3, 4])
    ocp.constraints.lbx = np.array(STATE_LBX)
    ocp.constraints.ubx = np.array(STATE_UBX)
    ocp.constraints.idxbu = np.array([0, 1, 2])
    ocp.constraints.lbu = np.array([CTRL_A_MIN, CTRL_W_MIN, CTRL_T_MIN])
    ocp.constraints.ubu = np.array([CTRL_A_MAX, CTRL_W_MAX, CTRL_T_MAX])

    x0 = np.array([0, 0, 0, 0, 0])
    ocp.constraints.x0 = x0

    ocp.model = model

    ocp.solver_options.tf = SOLVER_BASE_TF
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_cond_N = QP_COND_N
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.levenberg_marquardt = LM_DAMPING
    ocp.solver_options.nlp_solver_max_iter = NLP_MAX_ITER
    ocp.solver_options.qp_solver_iter_max = QP_MAX_ITER
    # Tight tolerances are important here: loose defaults can return status=0
    # while still producing trajectories that cut corners near obstacles.
    ocp.solver_options.nlp_solver_tol_stat = NLP_TOL_STAT
    ocp.solver_options.nlp_solver_tol_eq = NLP_TOL_EQ
    ocp.solver_options.nlp_solver_tol_ineq = NLP_TOL_INEQ
    ocp.solver_options.nlp_solver_tol_comp = NLP_TOL_COMP
    ocp.solver_options.print_level = 0
    ocp.solver_options.model_external_shared_lib_dir = l4c_model.shared_lib_dir
    ocp.solver_options.model_external_shared_lib_name = l4c_model.name

    acados_solver = AcadosOcpSolver(ocp, json_file="acados_mpc_npfield.json")

    return acados_solver
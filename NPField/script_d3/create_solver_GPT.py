from acados_template import AcadosOcp, AcadosOcpSolver , AcadosModel
from robot_model_GPT import robot_model
import numpy as np
import l4casadi as l4c
import torch

def create_solver(model_path, embedding_values):

    device = torch.device("cuda")
    model_path.to(device)
    model , l4c_model = robot_model(model_path, embedding_values)

    # acados OCP handle
    ocp = AcadosOcp()
    N = 30
    ocp.dims.N = N

    # OCP dimensions
    nx = 5
    nu = 3
    ny = nx + nu

    # OCP costs
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.model.cost_y_expr = model.cost_y_expr
    ocp.model.cost_y_expr_e = model.cost_y_expr_e

    ######## set weights for cost function
    w_x = 0.1 
    w_y = 0.1
    w_v = 0.05 
    w_theta = 0.1
    w_time  = 0.01
    w_a = 0.01
    w_w = 0.01
    w_T = 0.01
    w_x_e = 30
    w_y_e = 30
    w_v_e = 0.001
    w_theta_e = 0.01 
    w_time_e = 0.01 
    W_obst = np.array([3000])

    W_x = np.array([w_x, w_y, w_v, w_theta, w_time , w_a, w_w , w_T])
    W = np.diag(np.concatenate([W_x,W_obst]))
    W_xe = np.array([w_x_e, w_y_e, w_v_e, w_theta_e , w_time_e])
    W_e = np.diag(np.concatenate([W_xe, W_obst]))

    ocp.cost.W = W
    ocp.cost.W_e = W_e   
    ocp.cost.yref =  np.zeros([ny+1])
    ocp.cost.yref_e = np.zeros([nx+1])
    
    ocp.constraints.idxbx = np.array([0, 1, 2, 3 , 4])
    ocp.constraints.lbx = np.array([-100, -100, 0, -100 , 0])
    ocp.constraints.ubx = np.array([ 100,  100, 1,  100 , 100])
    ocp.constraints.idxbu = np.array([0, 1 , 2])
    ocp.constraints.lbu = np.array([-0.35, -0.4 , 0.2])
    ocp.constraints.ubu = np.array([ 0.35,  0.4 , 10])

    x0 = np.array([0, 0, 0, 0 , 0])
    ocp.constraints.x0 = x0

    ocp.model = model

    ocp.solver_options.tf = 25
    ocp.solver_options.qp_solver =  'PARTIAL_CONDENSING_HPIPM' 
    ocp.solver_options.qp_solver_cond_N = 10
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.levenberg_marquardt = 3.0
    ocp.solver_options.nlp_solver_max_iter = 15
    ocp.solver_options.qp_solver_iter_max = 100
    ocp.solver_options.nlp_solver_tol_stat = 1e-1
    ocp.solver_options.nlp_solver_tol_eq = 1e-1
    ocp.solver_options.print_level = 0
    ocp.solver_options.model_external_shared_lib_dir = l4c_model.shared_lib_dir
    ocp.solver_options.model_external_shared_lib_name = l4c_model.name



    acados_solver = AcadosOcpSolver(ocp, json_file="acados_mpc_npfield.json")

    return acados_solver
from acados_template import AcadosOcp, AcadosOcpSolver
from robot_model import robot_model
import numpy as np
import time
import math


def Solver():
    # acados OCP handle
    model = robot_model()
    ocp = AcadosOcp()

    ocp.dims.N = 30

    # OCP dimensions
    nx = 5
    nu = 2
    ny = nx + nu
    n_obst = 0

    # OCP costs
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.model.cost_y_expr = model.cost_y_expr
    ocp.model.cost_y_expr_e = model.cost_y_expr_e

    # Set path constraints bound
    ocp.constraints.lh = np.zeros(50)
    ocp.constraints.uh = 100000*np.ones(50)
  #  ocp.constraints.idxsh = np.arange(40)

    w_x = 20
    w_y = 20
    w_v = 0.001
    w_theta = 0.01
    w_delta = 0.01
    w_xe = 100
    w_ye = 100
    w_ve = 0.01
    w_thetae = 0.01
    w_a = 0.001
    w_w = 0.001

    # State and input cost
    W_obst = np.zeros(n_obst)
    for i in range(n_obst):
        W_obst[i]=  10


    W_x = np.array([w_x, w_y, w_v, w_theta,w_delta, w_a, w_w])
    W = np.diag(np.concatenate([W_x]))
    ocp.cost.W = W
    W_xe = np.array([w_xe,w_ye,w_ve,w_thetae , w_delta])
    W_e = np.diag(np.concatenate([W_xe]))
    ocp.cost.W_e = W_e   
    ocp.cost.yref = np.zeros([ny])
    ocp.cost.yref_e = np.zeros([nx])


    x_max = 200
    x_min = -200
    y_max = 200
    y_min = -200
    v_max = 1  
    v_min = 0
    theta_max = 100
    theta_min = -100
    a_max = 0.1 
    a_min = -0.1 
    w_max = 0.25 
    w_min = -0.25 

    
    ocp.constraints.idxbx = np.array([0, 1, 2 , 3 , 4])
    ocp.constraints.lbx = np.array([x_min, y_min, v_min , theta_min,-0.5])
    ocp.constraints.ubx = np.array([x_max, y_max, v_max , theta_max,0.5])
    ocp.constraints.idxbu = np.array([0, 1])
    ocp.constraints.lbu = np.array([a_min, w_min ])
    ocp.constraints.ubu = np.array([a_max, w_max ])

    paramters_static = [100 , 100] * 50 # n_obst
    paramters = np.concatenate([paramters_static])

    ocp.parameter_values = paramters
                                
    x0 = np.array([0, 0, 0 , 0,0])
    ocp.constraints.x0 = x0

    ocp.model = model

    ocp.solver_options.tf = 25
    ocp.solver_options.qp_solver =  'PARTIAL_CONDENSING_HPIPM' 
    ocp.solver_options.qp_solver_cond_N = 30
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.levenberg_marquardt = 3.0
    ocp.solver_options.nlp_solver_max_iter = 50
    ocp.solver_options.qp_solver_iter_max = 100
    ocp.solver_options.nlp_solver_tol_stat = 1e-2
    ocp.solver_options.nlp_solver_tol_eq = 1e-1
    ocp.solver_options.print_level = 0

    acados_solver = AcadosOcpSolver(ocp, json_file="acados_solver.json")

    return acados_solver


Solver()
print("Acados solver for NMPC problem was generated")
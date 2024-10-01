from acados_template import AcadosOcp, AcadosOcpSolver , AcadosModel
import numpy as np
import l4casadi as l4c
from model_nn import Autoencoder_path
import torch

from casadi import SX, DM, vertcat, sin, cos, tan, exp, if_else, pi , atan , logic_and , sqrt , fabs , atan2 , MX

    
def robot_model(model_loaded):
    model_name = "robot_model"

    # State
    x = MX.sym('x') 
    y = MX.sym('y')   
    v = MX.sym('v')  
    theta = MX.sym('theta')
    t_point = MX.sym('t_point') 

    sym_x = vertcat(x, y, v ,theta, t_point)

    # Input
    a = MX.sym('a')
    w = MX.sym('w')
    T = MX.sym('T')
    sym_u = vertcat(a, w , T)

    # Derivative of the States
    x_dot = MX.sym('x_dot')
    y_dot = MX.sym('y_dot')
    v_dot = MX.sym('v_dot')
    theta_dot = MX.sym('theta_dot')
    t_point_dot = MX.sym('t_point_dot')
    
    x_dot = vertcat(x_dot, y_dot, v_dot, theta_dot , t_point_dot)

    ## Model of Robot
    f_expl = T * vertcat(   sym_x[2] * cos(sym_x[3]),
                        sym_x[2] * sin(sym_x[3]),
                        sym_u[0],
                        sym_u[1],
                        1)
    f_impl = x_dot - f_expl

    model = AcadosModel()

    print(model)

    l4c_model = l4c.L4CasADi(model_loaded, model_expects_batch_dim=True , name='y_expr',device='cuda')

    torch.cuda.empty_cache()

    print("l4model " , l4c_model)
    num_out_embeding = 676
    num_prediction_steps_obst = 10
    t_update_dynamic_obst = 0.5

    dummy_embeding = np.zeros(num_out_embeding)

    sym_p = MX.sym('sym_p',num_out_embeding)
    embeding = MX.sym('in',num_out_embeding)
    embeding = vertcat(sym_p[0:676])
    cost_obst = MX.sym('cost_obst')

    potential_l4c_at_embeding = MX.sym('pot')
    potential_l4c_at_embeding = l4c_model(vertcat(embeding,x,y,theta))

    cost_obst = potential_l4c_at_embeding[0][0]
   
    for j in range(num_prediction_steps_obst):
        cond_1 = t_point >= j * t_update_dynamic_obst
        cond_2 = t_point <  (j * t_update_dynamic_obst + t_update_dynamic_obst)
        cond_3 = logic_and(cond_1 , cond_2)
        cost_obst = if_else(cond_3 , potential_l4c_at_embeding[j][0] , cost_obst)
 
    model.cost_y_expr = vertcat(sym_x, sym_u , cost_obst)
    model.cost_y_expr_e = vertcat(sym_x, cost_obst)
    
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = sym_x
    model.xdot = x_dot
    model.u = sym_u
    model.p = sym_p
    model.name = "robot_model"

    return model , l4c_model
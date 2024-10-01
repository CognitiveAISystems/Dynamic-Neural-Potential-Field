from acados_template import AcadosModel
import casadi as cd

def robot_model():
    model = AcadosModel()

    model.name = "robot_model"

    # State
    x = cd.SX.sym('x') 
    y = cd.SX.sym('y')   
    v = cd.SX.sym('v')  
    theta = cd.SX.sym('theta') 
    delta = cd.SX.sym('delta')

    sym_x = cd.vertcat(x, y, v ,theta, delta)
    model.x = sym_x

    # Input
    a = cd.SX.sym('a')
    d_delta = cd.SX.sym('d_delta')

    sym_u = cd.vertcat(a, d_delta)
    model.u = sym_u

    # Derivative of the States
    x_dot = cd.SX.sym('x_dot')
    y_dot = cd.SX.sym('y_dot')
    theta_dot = cd.SX.sym('theta_dot')
    v_dot = cd.SX.sym('v_dot')
    delta_dot = cd.SX.sym('delta_dot')
    x_dot = cd.vertcat(x_dot, y_dot, v_dot, theta_dot, delta_dot)
    L = 0.5 

    model.xdot = x_dot
    

    ## Model of Robot
    f_expl = cd.vertcat(sym_x[2] * cd.cos(sym_x[3]),
                    sym_x[2] * cd.sin(sym_x[3]),
                    sym_u[0],
                    cd.tan(sym_x[4])*sym_x[2]/L,
                    sym_u[1])
    f_impl = x_dot - f_expl
 
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl


    num_static_obst = 50
    obst_param = cd.SX.sym('p', 2 * num_static_obst)
    sym_p = cd.vertcat(obst_param)

    model.p = sym_p

    x_obst = obst_param[0::2]
    y_obst = obst_param[1::2]


    # potential function for a static obstacles
    distance_to_obst = cd.SX.sym('obst_stat',num_static_obst)
    e = 6
    for i in range(num_static_obst): 

        distance_to_obst[i] = (x_obst[i]-x)**2 + (y_obst[i]-y)**2 - (0.6)**2 
        
    model.cost_y_expr = cd.vertcat(sym_x, sym_u)
    model.cost_y_expr_e = cd.vertcat(sym_x)  


    # Path constraints

    
    model.con_h_expr = cd.vertcat(distance_to_obst)
    print(model)
  
    return model

robot_model()
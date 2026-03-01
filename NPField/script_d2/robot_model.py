from acados_template import AcadosModel
import l4casadi as l4c
import torch

from casadi import vertcat, cos, sin, if_else, logic_and, MX, fmin, fmax
from mpc_params import TIME_STEPS, OBSTACLE_PRED_DT

    
def robot_model(model_loaded, embedding_values):
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

    if embedding_values is None:
        raise ValueError("embedding_values must be provided for L4CasADi input.")

    class _EmbeddingWrappedModel(torch.nn.Module):
        def __init__(self, base_model, embedding):
            super().__init__()
            self.base_model = base_model
            emb = torch.tensor(embedding, dtype=torch.float32).view(1, -1)
            self.register_buffer("embedding", emb)

        def forward(self, xytheta):
            xytheta = xytheta.reshape(-1, 3)
            emb = self.embedding.to(device=xytheta.device, dtype=xytheta.dtype)
            emb = emb.expand(xytheta.shape[0], -1)
            inp = torch.cat([emb, xytheta], dim=1)
            return self.base_model(inp)

    wrapped_model = _EmbeddingWrappedModel(model_loaded, embedding_values)
    l4c_model = l4c.L4CasADi(wrapped_model, name="y_expr", device="cuda")

    torch.cuda.empty_cache()

    potential_all = l4c_model(vertcat(x, y, theta))

    max_t_pred = TIME_STEPS * OBSTACLE_PRED_DT
    t_clamped = fmin(fmax(t_point, 0), max_t_pred)

    cost_obst = potential_all[TIME_STEPS - 1]
    for j in range(TIME_STEPS):
        cond = logic_and(
            t_clamped >= j * OBSTACLE_PRED_DT,
            t_clamped < (j + 1) * OBSTACLE_PRED_DT,
        )
        cost_obst = if_else(cond, potential_all[j], cost_obst)
 
    model.cost_y_expr = vertcat(sym_x, sym_u , cost_obst)
    model.cost_y_expr_e = vertcat(sym_x, cost_obst)
    
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = sym_x
    model.xdot = x_dot
    model.u = sym_u
    model.name = "robot_model"

    return model , l4c_model
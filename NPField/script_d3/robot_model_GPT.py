from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import l4casadi as l4c
from model_nn_GPT import GPT
import torch

from casadi import (
    SX,
    vertcat,
    sin,
    cos,
    tan,
    exp,
    if_else,
    pi,
    atan,
    logic_and,
    sqrt,
    fabs,
    atan2,
    MX,
    fmin,
    fmax,
)
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

    num_prediction_steps_obst = TIME_STEPS
    t_update_dynamic_obst = OBSTACLE_PRED_DT
    cost_obst = MX.sym('cost_obst')


    potential_l4c_at_embeding = MX.sym('pot')
    potential_l4c_at_embeding = l4c_model(vertcat(x, y, theta))

    def _obstacle_potential_cost(step_idx: int):
        # The new D3 model predicts normalized potential in [0,1].
        # Keep it bounded and slightly sharpen high-risk regions.
        raw = potential_l4c_at_embeding[step_idx]
        raw = fmin(fmax(raw, 0.0), 1.0)
        return raw + 0.5 * raw * raw

    # Evaluate predicted obstacle potentials on [0, TIME_STEPS * OBSTACLE_PRED_DT]
    # and hold the last prediction afterwards.
    max_t_pred = num_prediction_steps_obst * t_update_dynamic_obst
    t_point_clamped = fmin(fmax(t_point, 0), max_t_pred)

    cost_obst = _obstacle_potential_cost(num_prediction_steps_obst - 1)
   
    for j in range(num_prediction_steps_obst):
        cond_1 = t_point_clamped >= j * t_update_dynamic_obst
        cond_2 = t_point_clamped < ((j + 1) * t_update_dynamic_obst)
        cond_3 = logic_and(cond_1, cond_2)
        cost_obst = if_else(cond_3, _obstacle_potential_cost(j), cost_obst)
 
    model.cost_y_expr = vertcat(sym_x, sym_u , cost_obst)
    model.cost_y_expr_e = vertcat(sym_x, cost_obst)
    
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = sym_x
    model.xdot = x_dot
    model.u = sym_u
    model.name = "robot_model"

    return model , l4c_model
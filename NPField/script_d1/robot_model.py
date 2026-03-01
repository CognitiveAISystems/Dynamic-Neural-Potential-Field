from acados_template import AcadosModel
import numpy as np
import l4casadi as l4c
import torch

from casadi import vertcat, if_else, logic_and, MX, fmin, fmax

from mpc_params import TIME_STEPS, OBSTACLE_PRED_DT


class _MultiFrameWrappedModel(torch.nn.Module):
    """Wraps a D1 Autoencoder_path so that a single (x, y, theta) query
    returns TIME_STEPS potential values — one per static-frame embedding.

    The base model's forward() expects (batch, 615) = 612 embedding + 3 coords
    and returns (batch, 1).  We batch all TIME_STEPS embeddings in one call
    and reshape to (batch, TIME_STEPS).
    """

    def __init__(self, base_model, all_embeddings):
        super().__init__()
        self.base_model = base_model
        self.register_buffer(
            "embeddings",
            torch.tensor(all_embeddings, dtype=torch.float32),
        )
        self.n_frames = all_embeddings.shape[0]
        self.eval()

    def forward(self, xytheta):
        xytheta = xytheta.reshape(-1, 3)
        batch_size = xytheta.shape[0]
        xytheta_rep = xytheta.repeat(self.n_frames, 1)
        emb = self.embeddings.to(device=xytheta.device, dtype=xytheta.dtype)
        emb_rep = emb.repeat_interleave(batch_size, dim=0)
        inp = torch.cat([emb_rep, xytheta_rep], dim=1)
        out = self.base_model(inp)
        return out.reshape(self.n_frames, batch_size).T


def robot_model(model_loaded, all_embeddings):
    """Build the acados CasADi model for D1.

    Parameters
    ----------
    model_loaded : Autoencoder_path
        Pre-trained D1 neural potential model.
    all_embeddings : ndarray (TIME_STEPS, 612)
        Pre-computed embeddings for each static sub-map frame.
    """
    x = MX.sym("x")
    y = MX.sym("y")
    v = MX.sym("v")
    theta = MX.sym("theta")
    t_point = MX.sym("t_point")
    sym_x = vertcat(x, y, v, theta, t_point)

    a = MX.sym("a")
    w = MX.sym("w")
    T = MX.sym("T")
    sym_u = vertcat(a, w, T)

    x_dot = MX.sym("x_dot")
    y_dot = MX.sym("y_dot")
    v_dot = MX.sym("v_dot")
    theta_dot = MX.sym("theta_dot")
    t_point_dot = MX.sym("t_point_dot")
    x_dot = vertcat(x_dot, y_dot, v_dot, theta_dot, t_point_dot)

    f_expl = T * vertcat(
        sym_x[2] * MX.cos(sym_x[3]),
        sym_x[2] * MX.sin(sym_x[3]),
        sym_u[0],
        sym_u[1],
        1,
    )
    f_impl = x_dot - f_expl

    model = AcadosModel()

    wrapped = _MultiFrameWrappedModel(model_loaded, all_embeddings)
    l4c_model = l4c.L4CasADi(wrapped, name="y_expr", device="cuda")
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

    model.cost_y_expr = vertcat(sym_x, sym_u, cost_obst)
    model.cost_y_expr_e = vertcat(sym_x, cost_obst)

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = sym_x
    model.xdot = x_dot
    model.u = sym_u
    model.name = "robot_model"

    return model, l4c_model

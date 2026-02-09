import argparse
import math
import os
import time
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pickle
import torch
from matplotlib import colors
from math import cos, sin

import create_solver_GPT as create_solver
from generate_MPC_config import generate_config
from model_nn_GPT import GPTConfig, GPT

N_HORIZON = 30
V_MAX = 0.6
TIME_STEPS = 10
POTENTIAL_RESOLUTION = 100


def resolve_paths() -> tuple[Path, Path, Path]:
    script_dir = Path(__file__).resolve().parent
    npfield_dir = script_dir.parent
    repo_root = npfield_dir.parent
    dataset_root = Path(
        os.getenv("NPFIELD_DATASET_DIR", repo_root / "NPField" / "dataset" / "dataset1000")
    )
    checkpoint_path = Path(
        os.getenv(
            "NPFIELD_CHECKPOINT",
            repo_root / "NPField" / "dataset" / "trained-models" / "NPField_onlyGPT_predmap9.pth",
        )
    )
    output_dir = npfield_dir / "output"
    return dataset_root, checkpoint_path, output_dir

def test_solver(acados_solver , x_ref_points , y_ref_points , theta_0 , num_map , ax1 , map_inp , dyn_obst_info):

    nx = 5
    nu = 3
    ny = nx + nu
    N = N_HORIZON
    yref = np.zeros([N, ny + 1])

    theta_e = 0
    v_0 = 0
    v_e = 0

    x_ref = []
    y_ref = []
    theta = []
    theta_ref = []
    init_x = []
    init_y = []
    init_theta = []
    len_segments = []
    theta = np.append(theta , theta_0 ) # current orientation robot
    theta_ref = np.append(theta_ref , theta_0 )
    num_segment = len(x_ref_points)-1
    length_path = 0
    for i in range(num_segment):
        length_path = length_path + math.sqrt((x_ref_points[i+1]-x_ref_points[i])**2+(y_ref_points[i+1]-y_ref_points[i])**2)
        theta = np.append(theta , math.atan2(y_ref_points[i+1]-y_ref_points[i], x_ref_points[i+1]-x_ref_points[i]))
        len_segments = np.append(len_segments , math.sqrt((x_ref_points[i+1]-x_ref_points[i])**2+(y_ref_points[i+1]-y_ref_points[i])**2))

    step_line = length_path / N

    print("length path",length_path)

    new_time_step = ((length_path) / (V_MAX * N)) * np.ones(N)

    k = 0
    x_ref = np.append(x_ref , x_ref_points[0])
    y_ref = np.append(y_ref , y_ref_points[0]) 
    for i in range(N+1):
        x_ref = np.append(x_ref , x_ref[i] + step_line * math.cos(theta[k+1]))
        y_ref = np.append(y_ref , y_ref[i] + step_line * math.sin(theta[k+1]))
        theta_ref = np.append(theta_ref , theta[k+1])
        d = math.sqrt((x_ref[-1]-x_ref_points[k])**2+(y_ref[-1]-y_ref_points[k])**2)
        if(d>len_segments[k] and k<(num_segment-1)):
            k = k+1
            x_ref[i] = x_ref_points[k]
            y_ref[i] = y_ref_points[k]
        elif (k>(num_segment-1)):
            break
    x0 = np.array([x_ref_points[0],y_ref_points[0],v_0,theta_0 , 0])

    init_x = x_ref[0:N+1]
    init_y = y_ref[0:N+1]
    init_theta = theta_ref[0:N+1]
    x_goal = np.array([init_x[-1], init_y[-1], v_e, init_theta[-1], length_path / V_MAX])

    yref[:,0]=init_x[0:N]
    yref[:,1]=init_y[0:N]
    yref[:,2] = V_MAX
    yref[:,3] = init_theta[0:N]
    yref[:,4]= np.linspace(0, length_path / V_MAX, N)

    a = np.zeros(1)

    yref_e = np.concatenate([x_goal,a])  
    x_traj_init = np.transpose([ yref[:,0] , yref[:,1] , yref[:,2], yref[:,3], yref[:,4]])

    simX = np.zeros((N+1, 5))
    simU = np.zeros((N, nu))

    for i in range(N):
        acados_solver.set(i,'y_ref',yref[i])
        acados_solver.set(i, 'x', x_traj_init[i])
        acados_solver.set(i, 'u', np.array([0.0, 0.0 , 1]))
    acados_solver.set(N, 'y_ref', yref_e)
    acados_solver.set(N, 'x', x_goal)
    acados_solver.set(0,'lbx', x0)
    acados_solver.set(0,'ubx', x0)
    acados_solver.set_new_time_steps(new_time_step)

    t = time.perf_counter()
    status = acados_solver.solve()
    print("status", status)
    elapsed = 1000 * (time.perf_counter() - t)
    print(f"Trajectory solve time (ms): {elapsed:.2f}")
    ROB_x = np.zeros([N+1,9])
    ROB_y = np.zeros([N+1,9])
    for i in range(N + 1):
        x = acados_solver.get(i, "x")
        simX[i,0]=x[0]
        simX[i,1]=x[1]
        simX[i,2]=x[2]
        simX[i,3]=x[3]
        simX[i,4]=round(x[4],1)
        ROB_x[i,0] = simX[i,0] + 0.6 * cos(simX[i,3]-0.59)
        ROB_x[i,1] = simX[i,0] + 0.514 * cos(simX[i,3]-0.24)
        ROB_x[i,2] = simX[i,0] + 0.75 * cos(simX[i,3]-0.16)
        ROB_x[i,3] = simX[i,0] + 0.75 * cos(simX[i,3]+0.16)
        ROB_x[i,4] = simX[i,0] + 0.514 * cos(simX[i,3]+0.24)
        ROB_x[i,5] = simX[i,0] + 0.6 * cos(simX[i,3]+0.59)
        ROB_x[i,6] = simX[i,0] - 0.6 * cos(simX[i,3]-0.59)
        ROB_x[i,7] = simX[i,0] - 0.6 * cos(simX[i,3]+0.59)
        ROB_x[i,8] = simX[i,0] + 0.6 * cos(simX[i,3]-0.59)
        ROB_y[i,0] = simX[i,1] + 0.6 * sin(simX[i,3]-0.59)
        ROB_y[i,1] = simX[i,1] + 0.514 * sin(simX[i,3]-0.24)
        ROB_y[i,2] = simX[i,1] + 0.75 * sin(simX[i,3]-0.16)
        ROB_y[i,3] = simX[i,1] + 0.75 * sin(simX[i,3]+0.16)
        ROB_y[i,4] = simX[i,1] + 0.514 * sin(simX[i,3]+0.24)
        ROB_y[i,5] = simX[i,1] + 0.6 * sin(simX[i,3]+0.59)
        ROB_y[i,6] = simX[i,1] - 0.6 * sin(simX[i,3]-0.59)
        ROB_y[i,7] = simX[i,1] - 0.6 * sin(simX[i,3]+0.59)
        ROB_y[i,8] = simX[i,1] + 0.6 * sin(simX[i,3]-0.59)
    
    initial_path = np.zeros((N + 1, 3))
    
    for i in range(N + 1):
        initial_path[i,0] = init_x[i]
        initial_path[i,1] = init_y[i]
        initial_path[i,2] = init_theta[i]
       
    for i in range(N):
        u = acados_solver.get(i, "u")
        simU[i,:]=u
    cost = acados_solver.get_cost()
    acados_solver.print_statistics()
    print("cost", cost)
    
    if (num_map ==-1):
        ax1.plot(simX[:, 0], simX[:, 1] , linewidth=4 , marker='o')
        ax1.plot(init_x , init_y, marker='o')
        ax1.set_aspect('equal', 'box')
        ax1.plot([3.5,5.1,5.1,3.5,3.5],[1.52,1.52,3.12,3.12,1.52] , linewidth=2 )
        ax1.plot([1.8,2.6,2.6,1.8,1.8],[3.92,3.92,4.72,4.72,3.92] , linewidth=2 )
        ax1.plot([0.6,2,2,0.6,0.6],[1.12,1.12,2.52,2.52,1.12] , linewidth=2 )
        plt.xlabel("x")
        plt.ylabel("y")
    else:
        for i in range(N+1):
            simX[i, 0] = simX[i, 0] * 10
            simX[i, 1] = simX[i, 1] * 10
            init_x[i]  = init_x[i] * 10
            init_y[i]  = init_y[i] * 10
            ROB_x[i,0] = ROB_x[i,0] * 10
            ROB_x[i,1] = ROB_x[i,1] * 10
            ROB_x[i,2] = ROB_x[i,2] * 10
            ROB_x[i,3] = ROB_x[i,3] * 10
            ROB_x[i,4] = ROB_x[i,4] * 10
            ROB_x[i,5] = ROB_x[i,5] * 10
            ROB_x[i,6] = ROB_x[i,6] * 10
            ROB_x[i,7] = ROB_x[i,7] * 10
            ROB_x[i,8] = ROB_x[i,8] * 10
            ROB_y[i,0] = ROB_y[i,0] * 10
            ROB_y[i,1] = ROB_y[i,1] * 10
            ROB_y[i,2] = ROB_y[i,2] * 10
            ROB_y[i,3] = ROB_y[i,3] * 10
            ROB_y[i,4] = ROB_y[i,4] * 10
            ROB_y[i,5] = ROB_y[i,5] * 10
            ROB_y[i,6] = ROB_y[i,6] * 10
            ROB_y[i,7] = ROB_y[i,7] * 10
            ROB_y[i,8] = ROB_y[i,8] * 10

        ax1.plot(simX[:, 0], simX[:, 1] , linewidth=2 , label="NPField path" )#'NPField path')  # marker='o',
        ax1.plot(init_x , init_y, linestyle='dashed', linewidth=2 , label='Initial path')
        ax1.legend()
        scale_x = 10
        scale_y = 10
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_x))
        ax1.xaxis.set_major_formatter(ticks_x)

        ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_y))
        ax1.yaxis.set_major_formatter(ticks_y)

    colors = ['k','k','r','r','g','g','b','b','c','c','m','m','y','y','k','k','r','r','g','g','b','b','c','c','m','m','y','y','k','k']
    for i in range(0,N,2):
        ax1.plot([ROB_x[i,0],ROB_x[i,5],ROB_x[i,6],ROB_x[i,7],ROB_x[i,0]], [ROB_y[i,0],ROB_y[i,5],ROB_y[i,6],ROB_y[i,7],ROB_y[i,0]], color=colors[i]) #'k'
        ax1.text(i*2.5 , 1 , str(round(simX[i,4],1))+"  ,  ",color=colors[i])

    x_obst = dyn_obst_info["initial_position"][num_map,0]
    y_obst = dyn_obst_info["initial_position"][num_map,1]
    theta_obst = dyn_obst_info["initial_position"][num_map,2]
    OBST_x = np.zeros([10,4])
    OBST_y = np.zeros([10,4])
    OBST_x[0,0] = x_obst + 0.291 * cos(theta_obst -1.03)
    OBST_x[0,1] = x_obst + 0.291 * cos(theta_obst +1.03)
    OBST_x[0,2] = x_obst - 0.291 * cos(theta_obst-1.03)
    OBST_x[0,3] = x_obst - 0.291 * cos(theta_obst+1.03)
    OBST_y[0,0] = y_obst + 0.291 * sin(theta_obst-1.03)
    OBST_y[0,1] = y_obst + 0.291 * sin(theta_obst+1.03)
    OBST_y[0,2] = y_obst - 0.291 * sin(theta_obst-1.03)
    OBST_y[0,3] = y_obst - 0.291 * sin(theta_obst+1.03)
    ax1.plot([OBST_x[0,0]*10,OBST_x[0,1]*10,OBST_x[0,2]*10,OBST_x[0,3]*10,OBST_x[0,0]*10], [OBST_y[0,0]*10,OBST_y[0,1]*10,OBST_y[0,2]*10,OBST_y[0,3]*10,OBST_y[0,0]*10], color='k')
    colors = ['r','g','b','c','m','y','k','r','g']
    ax1.text(0 , 5 , str(0)+", ",color=colors[0])
    for i in range(9):
        x_obst = x_obst + 0.5 * 0.5 * cos(theta_obst)
        y_obst = y_obst + 0.5 * 0.5 * sin(theta_obst)
        OBST_x[i+1,0] = x_obst + 0.291 * cos(theta_obst -1.03)
        OBST_x[i+1,1] = x_obst + 0.291 * cos(theta_obst +1.03)
        OBST_x[i+1,2] = x_obst - 0.291 * cos(theta_obst-1.03)
        OBST_x[i+1,3] = x_obst - 0.291 * cos(theta_obst+1.03)
        OBST_y[i+1,0] = y_obst + 0.291 * sin(theta_obst-1.03)
        OBST_y[i+1,1] = y_obst + 0.291 * sin(theta_obst+1.03)
        OBST_y[i+1,2] = y_obst - 0.291 * sin(theta_obst-1.03)
        OBST_y[i+1,3] = y_obst - 0.291 * sin(theta_obst+1.03)
        ax1.plot([OBST_x[i+1,0]*10,OBST_x[i+1,1]*10,OBST_x[i+1,2]*10,OBST_x[i+1,3]*10,OBST_x[i+1,0]*10], [OBST_y[i+1,0]*10,OBST_y[i+1,1]*10,OBST_y[i+1,2]*10,OBST_y[i+1,3]*10,OBST_y[i+1,0]*10], color=colors[i])
        ax1.text(1.5+i*2.5 , 5 , str(round(0.5+i*0.5,1))+"  ,  ",color=colors[i])
    path_mpc = simX

    return path_mpc , elapsed , ROB_x , ROB_y

def gif_generate(path , ROB_x , ROB_y, num_map, id_map, map_data, dyn_obst_info, output_dir: Path):
    x_obst = dyn_obst_info["initial_position"][num_map,0]
    y_obst = dyn_obst_info["initial_position"][num_map,1]
    theta_obst = dyn_obst_info["initial_position"][num_map,2]
    robot_path_x = path[:, 0]
    robot_path_y = path[:, 1]
    obst_path_x = [x_obst * 10.0]
    obst_path_y = [y_obst * 10.0]
    for _ in range(9):
        x_obst = x_obst + 0.5 * 0.5 * cos(theta_obst)
        y_obst = y_obst + 0.5 * 0.5 * sin(theta_obst)
        obst_path_x.append(x_obst * 10.0)
        obst_path_y.append(y_obst * 10.0)
    frames = []
    t_obst = 0.0
    t_robot = 0.0
    k_obst = 0
    k_robot = 0
    

    OBST_x = np.zeros([4])
    OBST_y = np.zeros([4])
    x_obst = dyn_obst_info["initial_position"][num_map,0] + 0.5 * 0.5 * cos(theta_obst)
    y_obst = dyn_obst_info["initial_position"][num_map,1] + 0.5 * 0.5 * sin(theta_obst)
    OBST_x[0] = x_obst + 0.291 * cos(theta_obst -1.03)
    OBST_x[1] = x_obst + 0.291 * cos(theta_obst +1.03)
    OBST_x[2] = x_obst - 0.291 * cos(theta_obst-1.03)
    OBST_x[3] = x_obst - 0.291 * cos(theta_obst+1.03)
    OBST_y[0] = y_obst + 0.291 * sin(theta_obst-1.03)
    OBST_y[1] = y_obst + 0.291 * sin(theta_obst+1.03)
    OBST_y[2] = y_obst - 0.291 * sin(theta_obst-1.03)
    OBST_y[3] = y_obst - 0.291 * sin(theta_obst+1.03)
    for i in np.arange(0.0,15.0,0.05):
        cmap = colors.ListedColormap(['white' , 'black'])
        fig2, ax2 = plt.subplots(1,1,figsize=(5,5))
        ax2.pcolor(map_data[num_map][0][::-1],cmap=cmap,edgecolors='w', linewidths=0.1)
        ax2.plot(robot_path_x, robot_path_y, color='r', linewidth=1)
        ax2.plot(robot_path_x[0], robot_path_y[0], marker='x', color='r', markersize=6)
        ax2.plot(robot_path_x[-1], robot_path_y[-1], marker='o', color='r', markersize=4)
        ax2.plot(obst_path_x, obst_path_y, color='b', linewidth=1)
        ax2.plot(obst_path_x[0], obst_path_y[0], marker='x', color='b', markersize=6)
        ax2.plot(obst_path_x[-1], obst_path_y[-1], marker='o', color='b', markersize=4)
        ax2.plot([OBST_x[0]*10,OBST_x[1]*10,OBST_x[2]*10,OBST_x[3]*10,OBST_x[0]*10], [OBST_y[0]*10,OBST_y[1]*10,OBST_y[2]*10,OBST_y[3]*10,OBST_y[0]*10], color='k')
        ax2.plot([ROB_x[k_robot,0],ROB_x[k_robot,5],ROB_x[k_robot,6],ROB_x[k_robot,7],ROB_x[k_robot,0]], [ROB_y[k_robot,0],ROB_y[k_robot,5],ROB_y[k_robot,6],ROB_y[k_robot,7],ROB_y[k_robot,0]], color='k') #'k'
        if (i > path[k_robot,4] and k_robot<29):
            k_robot+=1
        if (round(i,1) == t_obst and k_obst < 9):
            x_obst = x_obst + 0.5 * 0.5 * cos(theta_obst)
            y_obst = y_obst + 0.5 * 0.5 * sin(theta_obst)
            OBST_x[0] = x_obst + 0.291 * cos(theta_obst -1.03)
            OBST_x[1] = x_obst + 0.291 * cos(theta_obst +1.03)
            OBST_x[2] = x_obst - 0.291 * cos(theta_obst-1.03)
            OBST_x[3] = x_obst - 0.291 * cos(theta_obst+1.03)
            OBST_y[0] = y_obst + 0.291 * sin(theta_obst-1.03)
            OBST_y[1] = y_obst + 0.291 * sin(theta_obst+1.03)
            OBST_y[2] = y_obst - 0.291 * sin(theta_obst-1.03)
            OBST_y[3] = y_obst - 0.291 * sin(theta_obst+1.03)
            t_obst = t_obst + 0.5
            k_obst+=1
        fig2.canvas.draw()
        data = np.asarray(fig2.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(data)
        plt.close(fig2)


    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"D3_MAP_ep{num_map}_ID_{id_map}.gif"
    imageio.mimsave(str(output_path), frames, format="GIF", fps=20)

def fill_map_inp(num_map , map , footprint , obst_initial_position):
    map_inp = torch.zeros((10,5003))
    k = 0
    for n in range(10):
        for i in range (50):
            for j in range(50):
                map_inp[n][k] = map[num_map][n+1][i,j]  #[n+1][i,j] , [0][i,j]
                k = k + 1
        k = 0


    
    k = 0
    for n in range(10):
        for i in range (50):
            for j in range(50):
                map_inp[n][2500+k] = footprint[i,j]
                k = k +1
        k = 0

    map_inp[:,:5000]/=100.

    for i in range(10):
        map_inp[i,-3] = obst_initial_position[i][0]
        map_inp[i,-2] = obst_initial_position[i][1]
        map_inp[i,-1] = obst_initial_position[i][2]
    return map_inp.cuda()


###### SIMULATE #######

_SOLVER_CACHE = {}
_MAP_INP_CACHE = {}


def compute_embedding(model_loaded, map_inp):
    result = model_loaded.encode_map_footprint(map_inp[1]).detach()
    return result[0].detach().cpu().numpy()


def encode_map_for_potential(model_loaded, map_inp_row):
    if map_inp_row.dim() == 1:
        map_inp_row = map_inp_row.unsqueeze(0)
    with torch.no_grad():
        return model_loaded.encode_map_footprint(map_inp_row)


def infer_potential_grid(model_loaded, encoded, angle, device, chunk_size):
    xs = np.linspace(0.0, 5.0, POTENTIAL_RESOLUTION, endpoint=False)
    ys = np.linspace(0.0, 5.0, POTENTIAL_RESOLUTION, endpoint=False)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    coords = np.stack((grid_x.ravel(), grid_y.ravel()), axis=1)

    coords_t = torch.tensor(coords, dtype=torch.float32, device=device)
    theta = torch.full((coords_t.shape[0], 1), angle, dtype=torch.float32, device=device)
    encoded_rep = encoded.repeat(coords_t.shape[0], 1)
    input_batch = torch.hstack((encoded_rep, coords_t, theta))

    outputs = []
    with torch.no_grad():
        for start in range(0, input_batch.shape[0], chunk_size):
            chunk = input_batch[start : start + chunk_size]
            outputs.append(model_loaded(chunk).cpu())
    output = torch.cat(outputs, dim=0)
    return output.reshape(POTENTIAL_RESOLUTION, POTENTIAL_RESOLUTION, TIME_STEPS).permute(2, 0, 1).numpy()


def save_potential_gif(
    model_loaded,
    map_data,
    footprint,
    map_inp,
    num_map,
    id_map,
    theta_0,
    output_dir,
    chunk_size,
):
    device = map_inp.device.type
    encoded = encode_map_for_potential(model_loaded, map_inp[1])
    res_array = infer_potential_grid(model_loaded, encoded, theta_0, device, chunk_size)

    frames = []
    for tme in range(TIME_STEPS):
        fig, (ax1, ax2) = plt.subplots(
            nrows=1, ncols=2, figsize=(12, 5), gridspec_kw={"wspace": 0.3, "hspace": 0.1}
        )
        ax1.imshow(map_data[num_map][tme])
        ax2.imshow(np.rot90(res_array[tme]))
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(frame)
        plt.close(fig)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"NPField_D3_potential_ep{num_map}_ID_{id_map}.gif"
    imageio.mimsave(str(output_path), frames, format="GIF", loop=65535)


def get_acados_solver(model_loaded, map_inp, cache_key):
    if cache_key in _SOLVER_CACHE:
        return _SOLVER_CACHE[cache_key]
    embedding_values = compute_embedding(model_loaded, map_inp)
    acados_solver = create_solver.create_solver(model_loaded, embedding_values)
    _SOLVER_CACHE[cache_key] = acados_solver
    return acados_solver

def run_simulation(
    num_map,
    x_ref_points,
    y_ref_points,
    theta_0,
    obst_motion_info,
    map_data,
    footprint,
    model_loaded,
    dyn_obst_info,
    output_dir,
    save_potential_gif_flag,
    potential_chunk_size,
    id_map=0,
):
    id_dyn = 0
    obst_initial_position = obst_motion_info['motion_dynamic_obst'][num_map, id_dyn*10:(id_dyn+1)*10]

    cache_key = (num_map, id_dyn)
    if cache_key in _MAP_INP_CACHE:
        map_inp = _MAP_INP_CACHE[cache_key]
    else:
        map_inp = fill_map_inp(num_map, map_data, footprint, obst_initial_position)
        _MAP_INP_CACHE[cache_key] = map_inp
    acados_solver = get_acados_solver(model_loaded, map_inp, cache_key=cache_key)

    cmap = colors.ListedColormap(['white', 'black'])
    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_box_aspect(1)
    ax.pcolor(map_data[num_map][0][::-1], cmap=cmap, edgecolors='w', linewidths=0.1)

    path_mpc, elapsed, ROB_x, ROB_y = test_solver(
        acados_solver, x_ref_points, y_ref_points, theta_0, num_map, ax, map_inp, dyn_obst_info
    )
    print(f"Trajectory time without solver init (ms): {elapsed:.2f}")
    gif_generate(path_mpc, ROB_x, ROB_y, num_map, id_map, map_data, dyn_obst_info, output_dir)
    if save_potential_gif_flag:
        save_potential_gif(
            model_loaded=model_loaded,
            map_data=map_data,
            footprint=footprint,
            map_inp=map_inp,
            num_map=num_map,
            id_map=id_map,
            theta_0=theta_0,
            output_dir=output_dir,
            chunk_size=potential_chunk_size,
        )

    fig, ax2 = plt.subplots(1)
    ax2.plot(path_mpc[:,4], path_mpc[:,2])
    ax2.grid()
    ax2.set_ylim([0, 1.1])
    plt.setp(ax2, ylabel='v (m/sec)')
    plt.show(block=False)

    return path_mpc, elapsed  # You can return these if needed elsewhere


def load_datasets(dataset_root):
    sub_maps = pickle.load(open(dataset_root / "dataset_1000_maps_0_100_all.pkl", "rb"))
    d_footprint = pickle.load(open(dataset_root / "data_footprint.pkl", "rb"))
    dyn_obst_info = pickle.load(open(dataset_root / "dataset_initial_position_dynamic_obst.pkl", "rb"))
    obst_motion_info = pickle.load(open(dataset_root / "dataset_1000_maps_obst_motion.pkl", "rb"))
    costmaps = pickle.load(open(dataset_root / "dataset_1000_costmaps.pkl", "rb"))
    return (
        sub_maps["submaps"],
        d_footprint["footprint_husky"],
        dyn_obst_info,
        obst_motion_info,
        costmaps["costmap"],
    )


def load_model(checkpoint_path):
    model_args = dict(
        n_layer=4,
        n_head=4,
        n_embd=576,
        block_size=1024,
        bias=True,
        vocab_size=1024,
        dropout=0.1,
    )
    gptconf = GPTConfig(**model_args)
    model_loaded = GPT(gptconf)
    pretrained_dict = torch.load(checkpoint_path)
    model_dict = model_loaded.state_dict()
    rejected_keys = [k for k, v in model_dict.items() if k not in pretrained_dict]
    print("REJECTED KEYS: ", rejected_keys)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model_loaded.load_state_dict(model_dict)
    model_loaded.to(torch.device("cuda"))
    model_loaded.eval()
    return model_loaded


def parse_args():
    parser = argparse.ArgumentParser(description="Run NPField solver.")
    parser.add_argument("--map-id", type=int, default=3, help="Map index to use.")
    parser.add_argument("--num-orientation", type=int, default=0, help="Orientation index.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of runs.")
    parser.add_argument(
        "--save-potential-gif",
        action="store_true",
        help="Save potential field GIF for each run.",
    )
    parser.add_argument(
        "--potential-chunk-size",
        type=int,
        default=4096,
        help="Batch size for potential grid inference.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root, checkpoint_path, output_dir = resolve_paths()
    map_data, footprint, dyn_obst_info, obst_motion_info, costmap = load_datasets(dataset_root)
    model_loaded = load_model(checkpoint_path)

    for i in range(args.episodes):
        x_ref_points, y_ref_points, theta_0 = generate_config(costmap, args.map_id, args.num_orientation)
        run_simulation(
            num_map=args.map_id,
            x_ref_points=x_ref_points,
            y_ref_points=y_ref_points,
            theta_0=theta_0,
            obst_motion_info=obst_motion_info,
            map_data=map_data,
            footprint=footprint,
            model_loaded=model_loaded,
            dyn_obst_info=dyn_obst_info,
            output_dir=output_dir,
            save_potential_gif_flag=args.save_potential_gif,
            potential_chunk_size=args.potential_chunk_size,
            id_map=i,
        )


if __name__ == "__main__":
    main()


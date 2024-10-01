import torch
import json
from matplotlib import colors , transforms
import sys
import time
import json
import matplotlib.ticker as ticker
from json import JSONEncoder
import l4casadi as l4c
import imageio
import matplotlib.pyplot as plt
import pickle
import create_solver_GPT as create_solver
from math import cos , sin

import numpy as np
import math
from model_nn_GPT import GPTConfig, GPT 

def test_solver(acados_solver , x_ref_points , y_ref_points , theta_0 , num_map , ax1 , map_inp):

    nx = 5
    nu = 3
    ny = nx + nu
    N = 30
    yref = np.zeros([N,ny+1])

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

    v_max = 0.6

    new_time_step = ((length_path)/(v_max*N)) * np.ones(N)

    print(new_time_step)

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
    x_goal = np.array([init_x[-1],init_y[-1], v_e,init_theta[-1] , length_path/v_max])

    yref[:,0]=init_x[0:N]
    yref[:,1]=init_y[0:N]
    yref[:,2] = v_max
    yref[:,3] = init_theta[0:N]
    yref[:,4]= np.linspace(0,length_path/v_max,N)

    a = np.zeros(1)

    paramters_static = [100] * 3456 

    
    
    result = model_loaded.encode_map_footprint(map_inp[1]).detach()
    for i in range(3456):
        paramters_static[i] = result[0,i].detach().cpu().numpy()

    parameter_values = np.concatenate([paramters_static])

    yref_e = np.concatenate([x_goal,a])  
    x_traj_init = np.transpose([ yref[:,0] , yref[:,1] , yref[:,2], yref[:,3], yref[:,4]])

    simX = np.zeros((N+1, 5))
    simU = np.zeros((N, nu))

    for i in range(N):
        acados_solver.set(i,'p',parameter_values)
        acados_solver.set(i,'y_ref',yref[i])
        acados_solver.set(i, 'x', x_traj_init[i])
        acados_solver.set(i, 'u', np.array([0.0, 0.0 , 1]))
    acados_solver.set(N, 'p',  parameter_values)
    acados_solver.set(N, 'y_ref', yref_e)
    acados_solver.set(N, 'x', x_goal)
    acados_solver.set(0,'lbx', x0)
    acados_solver.set(0,'ubx', x0)
    acados_solver.options_set('rti_phase', 0)
    acados_solver.set_new_time_steps(new_time_step)

    t = time.time()
    status = acados_solver.solve()
    print("status" , status)
    elapsed = 1000 * (time.time() - t)
    print("Elapsed time: {} ms".format(elapsed))
    print("optimal path")
    ROB_x = np.zeros([N+1,9])
    ROB_y = np.zeros([N+1,9])
    for i in range(N + 1):
        x = acados_solver.get(i, "x")
        print(x[0],", ",x[1], ",",x[2],", ",x[3] ,", ",x[4])
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
    
    print("initial path")
    
    for i in range(N + 1):
        initial_path[i,0] = init_x[i]
        initial_path[i,1] = init_y[i]
        initial_path[i,2] = init_theta[i]
        print(initial_path[i,0],", ",initial_path[i,1] , ", ", initial_path[i,2])
       
    for i in range(N):
        u = acados_solver.get(i, "u")
        simU[i,:]=u
    print("status" , status)
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
    print(dyn_obst_info["initial_position"][num_map,0],dyn_obst_info["initial_position"][num_map,1])
    path_mpc = simX

    return path_mpc , parameter_values , elapsed , ROB_x , ROB_y

def gif_generate(path , ROB_x , ROB_y,num_map,id_map):
    x_obst = dyn_obst_info["initial_position"][num_map,0]
    y_obst = dyn_obst_info["initial_position"][num_map,1]
    theta_obst = dyn_obst_info["initial_position"][num_map,2]
    frames = []
    t_obst = 0.0
    t_robot = 0.0
    k_obst = 0
    k_robot = 0
    

    OBST_x = np.zeros([4])
    OBST_y = np.zeros([4])
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
    for i in np.arange(0.0,15.0,0.05):
        cmap = colors.ListedColormap(['white' , 'black'])
        fig2, ax2 = plt.subplots(1,1,figsize=(5,5))
        ax2.pcolor(map[num_map][0][::-1],cmap=cmap,edgecolors='w', linewidths=0.1)
        ax2.plot([OBST_x[0]*10,OBST_x[1]*10,OBST_x[2]*10,OBST_x[3]*10,OBST_x[0]*10], [OBST_y[0]*10,OBST_y[1]*10,OBST_y[2]*10,OBST_y[3]*10,OBST_y[0]*10], color='k')
        ax2.plot([ROB_x[k_robot,0],ROB_x[k_robot,5],ROB_x[k_robot,6],ROB_x[k_robot,7],ROB_x[k_robot,0]], [ROB_y[k_robot,0],ROB_y[k_robot,5],ROB_y[k_robot,6],ROB_y[k_robot,7],ROB_y[k_robot,0]], color='k') #'k'
        if (i > path[k_robot,4] and k_robot<29):
            k_robot+=1
        if (round(i,1) == t_obst and k_obst < 9):
            print("i" , i)
            print(x_obst , y_obst , theta_obst)
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
            print(OBST_x[0] , OBST_x[1])
            t_obst = t_obst + 0.5
            k_obst+=1
        fig2.canvas.draw()
        data = np.frombuffer(fig2.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig2.canvas.get_width_height()[::-1] + (3,))
        frames.append(data)
        plt.close(fig2)


    imageio.mimsave(f'../../outputs/D3_MAP_{str(num_map)}_ID_{str(id_map)}.gif', frames, format="GIF", fps=20)  

def fill_map_inp(num_map , map , footprint , obst_initial_position):
    print(map.shape)
    print(footprint.shape)
    map_inp = torch.zeros((10,5003))
    k = 0
    for n in range(10):
        for i in range (50):
            for j in range(50):
                map_inp[n][k] = map[num_map][n+1][i,j]  #[n+1][i,j] , [0][i,j]
        #         if(map_inp[n][k] == 100):
        #             map_inp[n][k] = 1
                k = k + 1
        k = 0
    
    k = 0
    for n in range(10):
        for i in range (50):
            for j in range(50):
                map_inp[n][2500+k] = footprint[i,j]
        #         if(map_inp[n][2500+k] == 100):
        #             map_inp[n][2500+k] = 1
                k = k +1
        k = 0

    map_inp[:,:5000]/=100.

    for i in range(10):
        map_inp[i,-3] = obst_initial_position[i][0]
        map_inp[i,-2] = obst_initial_position[i][1]
        map_inp[i,-1] = obst_initial_position[i][2]
    return map_inp.cuda()


###### SIMULATE #######

def run_simulation(num_map, x_ref_points, y_ref_points, theta_0, obst_motion_info, map, footprint, acados_solver,id_map=0):
    id_dyn = 0
    obst_initial_position = obst_motion_info['motion_dynamic_obst'][num_map, id_dyn*10:(id_dyn+1)*10]
    print("obst map", num_map, "is", obst_initial_position)

    map_inp = fill_map_inp(num_map, map, footprint, obst_initial_position)
    print("map shape", map_inp.shape)

    cmap = colors.ListedColormap(['white', 'black'])
    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_box_aspect(1)
    ax.pcolor(map[num_map][0][::-1], cmap=cmap, edgecolors='w', linewidths=0.1)

    path_mpc, parameters, elapsed, ROB_x, ROB_y = test_solver(acados_solver, x_ref_points, y_ref_points, theta_0, num_map, ax, map_inp)
    gif_generate(path_mpc, ROB_x, ROB_y, num_map,id_map)

    fig, ax2 = plt.subplots(1)
    ax2.plot(path_mpc[:,4], path_mpc[:,2])
    ax2.grid()
    ax2.set_ylim([0, 1.1])
    plt.setp(ax2, ylabel='v (m/sec)')
    plt.show(block=False)

    return path_mpc, parameters, elapsed  # You can return these if needed elsewhere


############## Load Datasets to test the created solver ###########
sub_maps = pickle.load(open("../../dataset/dataset1000/dataset_1000_maps_0_100_all.pkl", "rb"))
d_footprint = pickle.load(open("../../dataset/dataset1000/data_footprint.pkl", "rb"))
dyn_obst_info = pickle.load(open("../../dataset/dataset1000/dataset_initial_position_dynamic_obst.pkl", "rb"))
obst_motion_info = pickle.load(open("../../dataset/dataset1000/dataset_1000_maps_obst_motion.pkl", "rb"))


map = sub_maps['submaps']
footprint = d_footprint['footprint_husky']


######## load CNN Model
n_layer = 4
n_head = 4
n_embd = 576
dropout = 0.1 # for pretraining 0 is good, for finetuning try 0.1+
bias = True #False
block_size = 1024
device = torch.device("cuda")

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)
model_args['vocab_size'] = 1024 # 50257 2048

gptconf = GPTConfig(**model_args)

model_loaded = GPT(gptconf)
checkpoint_name = '../../trained-models/NPField_onlyGPT_predmap9.pth'



pretrained_dict = torch.load(checkpoint_name)
model_dict = model_loaded.state_dict()
# 1. filter out unnecessary keys
rejected_keys = [k for k, v in model_dict.items() if k not in pretrained_dict]
print('REJECTED KEYS: ',rejected_keys)
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict) 
# 3. load the new state dict
model_loaded.load_state_dict(model_dict)


"""
sd = model_loaded.state_dict()
sd_keys = sd.keys()
sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param
sd_hf = torch.load(checkpoint_name)

sd_keys_hf = sd_hf.keys()
sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
# basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
# this means that we have to transpose these weights when we import them
#assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
for k in sd_keys_hf:
    if any(k.endswith(w) for w in transposed):
        # special treatment for the Conv1D weights we need to transpose
        assert sd_hf[k].shape[::-1] == sd[k].shape
        with torch.no_grad():
            sd[k].copy_(sd_hf[k].t())
    else:
        # vanilla copy over the other parameters
        assert sd_hf[k].shape == sd[k].shape
        with torch.no_grad():
            sd[k].copy_(sd_hf[k])
"""



#load_check = torch.load(checkpoint_name)
#model_loaded.load_state_dict(load_check)
model_loaded.to(device);
model_loaded.eval();

##### create solver
acados_solver = create_solver.create_solver(model_loaded)

###################### Reference path #######################

N = 30
path_mpc = np.zeros((N+1, 4))
initial_path = np.zeros((N+1 , 3))

from generate_MPC_config import generate_config
costmaps = pickle.load(open("../../dataset/dataset1000/dataset_1000_costmaps.pkl", "rb"))

num_map = 3
num_orientation = 0


for i in range(10):
    x_ref_points, y_ref_points, theta_0 = generate_config(costmaps['costmap'], num_map, num_orientation)

    run_simulation(
        num_map=num_map,
        x_ref_points=x_ref_points,
        y_ref_points=y_ref_points,
        theta_0=theta_0,
        obst_motion_info=obst_motion_info,
        map=map,
        footprint=footprint,
        acados_solver=acados_solver,
        id_map = i
    )


from robot_model import robot_model
import matplotlib.pyplot as plt
import numpy as np
import time
import math
import create_solver

from ctypes import sizeof
#import json
#from json import JSONEncoder
#fig, ax1 = plt.subplots(1,2)

    
def store_data_path(path ,name_file, x_ref_points ,y_ref_points, theta_0 , theta_e , elapsed_time):
    # Serialization
    start_point = [x_ref_points[0] , y_ref_points[0] , theta_0]
    stop_point = [x_ref_points[-1] , y_ref_points[-1] , theta_e]
    numpyData = {"path_ciao": path, "start_point": start_point,"stop_point": stop_point, "cal_time": elapsed_time}
    # Deserialization
    print("Decode JSON serialized NumPy array")
  #  print("NumPy Array")
  #  print(finalNumpyArray)


def test_solver(x_ref_points , y_ref_points , theta_initial , paramters_static):
    acados_solver = create_solver.Solver()
    print("START SOLVING")
    nx = 5
    nu = 2
    ny = nx + nu
    N = 30
    yref = np.zeros([N,ny])
    theta_0 = theta_initial
    v_0 = 0
    
    v_e = 0

    x_ref = []
    y_ref = []
    theta_segments = []
    theta_ref = []
    init_x = []
    init_y = []
    init_theta = []
    N = 30
    len_segments = []
    theta_segments = np.append(theta_segments , theta_0 ) # current orientation robot
    theta_ref = np.append(theta_ref , theta_0 )
    num_segment = len(x_ref_points)-1
    length_path = 0
    for i in range(num_segment):
        length_path = length_path + math.sqrt((x_ref_points[i+1]-x_ref_points[i])**2+(y_ref_points[i+1]-y_ref_points[i])**2)
        theta_middle = math.atan2(y_ref_points[i+1]-y_ref_points[i], x_ref_points[i+1]-x_ref_points[i])
        if(theta_0>1.57 and theta_middle<0):
          theta_middle = 6.28 + theta_middle
        theta_segments = np.append(theta_segments , theta_middle)
        x_center_segment = (x_ref_points[i+1]+x_ref_points[i])/2
        y_center_segment = (y_ref_points[i+1]+y_ref_points[i])/2 
        print("segment center" , x_center_segment , y_center_segment)
        len_segments = np.append(len_segments , math.sqrt((x_ref_points[i+1]-x_ref_points[i])**2+(y_ref_points[i+1]-y_ref_points[i])**2))
      #  my_python_utils.plot_high_order_ellipse(ax1 , 8 , x_center_segment  , y_center_segment, (len_segments[i]/2)+0.5, 1.5 , theta_middle)

    step_line = length_path / N
    theta_e = theta_segments[-1] #1.57 #theta[1]
    print("theta_e",theta_e)
    print(length_path)

    k = 0
    x_ref = np.append(x_ref , x_ref_points[0])
    y_ref = np.append(y_ref , y_ref_points[0])
    for i in range(N+1):
        x_ref = np.append(x_ref , x_ref[i] + step_line * math.cos(theta_segments[k+1]))
        y_ref = np.append(y_ref , y_ref[i] + step_line * math.sin(theta_segments[k+1]))
        theta_ref = np.append(theta_ref , theta_segments[k+1])
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
    x_goal = np.array([init_x[-1],init_y[-1],v_e,theta_e , 0])
    print("x_goal" , x_goal)
        
    v_d = 0.2

    new_time_step = ((length_path)/(v_d*N)) * np.ones(N)





    parameter_values = np.concatenate([paramters_static])
    print("Parameter Values" , parameter_values)
  #  for i in range(4*num_edge):
  #      print(parameter_values[i])

    yref[:,0]=init_x[0:N]
    yref[:,1]=init_y[0:N]
    yref[:,2]=0.2
    yref[:,3] = init_theta[0:N]
    yref[:,4] = 0
    print(init_theta)
    yref_e = np.concatenate([x_goal])  
    x_traj_init = np.transpose([ yref[:,0] , yref[:,1] , yref[:,2] , yref[:,3] , yref[:,4]])

    for i in range(30):
        print("YREF" , yref[i,0],yref[i,1],yref[i,3])

    simX = np.zeros((N+1, 5))
    simU = np.zeros((N, 2))
 
    #acados_solver.constraints.x0 = x0
    for i in range(N):
        acados_solver.set(i,'p',parameter_values)
        acados_solver.set(i,'y_ref',yref[i])
        acados_solver.set(i, 'x', x_traj_init[i])
        acados_solver.set(i, 'u', np.array([0.0, 0.0]))
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
    elapsed = time.time() - t
    for i in range(N + 1):
        x = acados_solver.get(i, "x")
        simX[i,:]=x
    cost = acados_solver.get_cost()
    print("cost" , cost)    
    for i in range(N):
        u = acados_solver.get(i, "u")
        simU[i,:]=u

   
    length_gotten_path = 0
    for i in range(1,N+1):
        length_gotten_path = length_gotten_path + math.sqrt((simX[i,0]-simX[i-1,0])**2+(simX[i,1]-simX[i-1,1])**2)

        
    print("Elapsed time: {} ms".format(elapsed) , "length_gotten_path: {} m".format(length_gotten_path), "length_initial: {} m".format(length_path))
 
    
 #   

    
  #  for i in range(0,len(x_edge),4):
  #      ax1.plot([x_edge[i] , x_edge[i+1], x_edge[i+2], x_edge[i+3]], [y_edge[i] , y_edge[i+1], y_edge[i+2], y_edge[i+3]], color='g')
  #      ax1.fill_between([x_edge[i] , x_edge[i+1], x_edge[i+2], x_edge[i+3]], [y_edge[i] , y_edge[i+1], y_edge[i+2], y_edge[i+3]], color='C0', alpha=0.3)
  #  ax1.plot([x_edge[0] , x_edge[1]], [y_edge[0] , y_edge[1]], color='g', label='Far polygons')

    # for i in range(0,len(x_edge_import),2):
    #     ax1.plot([x_edge_import[i] , x_edge_import[i+1]], [y_edge_import[i] , y_edge_import[i+1]], color='r')
    # ax1.plot([x_edge_import[0] , x_edge_import[1]], [y_edge_import[0] , y_edge_import[1]], color='r', label='Important polygons')

    print(simX[:, 0])
    
 #   ax1.plot(simX[:, 0], simX[:, 1] , linewidth=4)
 #   ax1.plot(init_x , init_y, marker='o')

   # plt.grid()
  #  plt.xlabel("x")
  #  plt.ylabel("y")
    return simX , init_x , init_y , theta_e , elapsed


times_calc = np.zeros(1)
sum_times = 0

fig, ax1 = plt.subplots(1)

fig.tight_layout()


############## taken case number 1 
x_ref_points = [[8.9 , 11],[3.45 , 8.27], [8 , 11.13], [3.8 , 13] , [11.7 , 7.6], [0.88 , 7.26], [7.07 , 5.2] , [6 ,  12.2 ] , [7.3 , 6] , [0.6 , 7.86]]
y_ref_points = [[7 , 11.45], [6.37 , 5.04], [7.6 , 1], [6.3 , 11.2] , [1.5 , 6.4], [6.58 , 9.93], [2.93 , 7.4], [7.55, 11.2] , [1.9 , 7.55] , [6.6 , 10.35]]
theta_start = [1.57 , 0 , -0.9 , 0 , 2 , 0 , 1.57, 0 , 0 ,-0.7]
k = [90 , 0 , 90 , 90 , 90 , 40 , 0, 90 , 0 , 40]
num_case = 8
x_ref_points = [x_ref_points[num_case]]
y_ref_points = [y_ref_points[num_case]]
k = k[num_case]

simplot2 = np.zeros((len(x_ref_points)*31, 2))
siminitial2 = np.zeros((len(x_ref_points)*31, 2))
theta_initial = theta_start[num_case]

paramters_static = [100 , 100] * 50
## map 2
x_edge = np.array([ 6.1656,0.38585,0.38585,0.74735,0.74735,5.85675,5.85675,6.1656,     3.94143, 7.8314,7.8314,   1.90134,1.90134,  -0.91932,-0.91932, 0.86538,0.86538, 3.94143,  10.37266, 9.64212,9.64212, 12.84587,12.84587, 15.11149,15.11149, 14.79137,14.79137, 10.37266])
y_edge = np.array([5.47785,2.32167,2.32167,-0.35042,-0.35042,0.03288,0.03288,5.47785,   8.36263, 11.84857,11.84857, 13.02051, 13.02051,  7.57639,7.57639, 7.80464,7.80464, 8.36263,   9.59585, 4.03978,4.03978,  2.31113,2.31113,  4.14933,4.14933,  7.15361,7.15361,  9.59585])

x_obst = []
y_obst = []
for i in range(0,len(x_edge),2):
    x_obst_ = np.linspace(x_edge[i] , x_edge[i+1] , 10)
    y_obst_ = np.linspace(y_edge[i] , y_edge[i+1] , 10)
    x_obst = np.append(x_obst , x_obst_)
    y_obst = np.append(y_obst , y_obst_)



for i in range(0, 100 , 2):
    paramters_static[i] = x_obst[k]
    paramters_static[i+1] = y_obst[k]
    k = k +1

for i in range(0,100,2):
    circle1 = plt.Circle((paramters_static[i], paramters_static[i+1]),  0.1, color='r')
    ax1.add_patch(circle1)

ROB_x = np.zeros([31,5])
ROB_y = np.zeros([31,5])
from math import cos , sin
for i in range(len(x_ref_points)):
    Simx , init_x , init_y ,theta_e , elapsed_time = test_solver(x_ref_points[i] , y_ref_points[i], theta_initial , paramters_static)
    for j in range(31):
        simplot2[i*31 + j,0] =  Simx[j,0]
        simplot2[i*31 + j,1] =  Simx[j,1]
        siminitial2[i*31 + j,0] =  init_x[j]
        siminitial2[i*31 + j,1] =  init_y[j]
name_file = "ciao_test"+str(num_case)+".json"
store_data_path(Simx ,name_file, init_x ,init_y, theta_initial , theta_e , elapsed_time)
for i in range(31):
    ROB_x[i,0] = Simx[i,0] + 0.6 * cos(Simx[i,3]-0.6)
    ROB_x[i,1] = Simx[i,0] + 0.6 * cos(Simx[i,3]+0.6)
    ROB_x[i,2] = Simx[i,0] - 0.6 * cos(Simx[i,3]-0.6)
    ROB_x[i,3] = Simx[i,0] - 0.6 * cos(Simx[i,3]+0.6)
    ROB_x[i,4] = Simx[i,0] + 0.6 * cos(Simx[i,3]-0.6)
    ROB_y[i,0] = Simx[i,1] + 0.6 * sin(Simx[i,3]-0.6)
    ROB_y[i,1] = Simx[i,1] + 0.6 * sin(Simx[i,3]+0.6)
    ROB_y[i,2] = Simx[i,1] - 0.6 * sin(Simx[i,3]-0.6)
    ROB_y[i,3] = Simx[i,1] - 0.6 * sin(Simx[i,3]+0.6)
    ROB_y[i,4] = Simx[i,1] + 0.6 * sin(Simx[i,3]-0.6)
 
ax1.plot(simplot2[:, 0], simplot2[:, 1] , linewidth=2, label='CIAO Path')#,marker='o')
ax1.plot(siminitial2[:, 0], siminitial2[:, 1] , linewidth=2, linestyle='dashed', label='Initial Path')
for i in range(31):
    ax1.plot([ROB_x[i,0],ROB_x[i,1],ROB_x[i,2],ROB_x[i,3],ROB_x[i,0]], [ROB_y[i,0],ROB_y[i,1],ROB_y[i,2],ROB_y[i,3],ROB_y[i,0]],linestyle='dashed', color='k')
      


for i in range(0,len(x_edge),2):
    ax1.plot([x_edge[i] , x_edge[i+1]], [y_edge[i] , y_edge[i+1]], color='g')
ax1.legend()

plt.show()

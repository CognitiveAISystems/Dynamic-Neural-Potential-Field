import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
from math import floor, sqrt, atan2



def generate_config(costmaps, num_map, num_orientation):

    l_first = []
    l_end = []
    coord = []
    x_path = []
    y_path = []
    num_orientation = random.randint(0,2)

    for i in range(2000):
        x = np.random.uniform(0.8, 4.2) 
        y = np.random.uniform(0.8, 4.2) 
        i = round((5 - y)/0.10)
        j = round(x/0.10)
        if(costmaps[num_map][(num_orientation*10)+1][i][j]>1.8):
            l_first.append(x)
            l_first.append(y)
    for i in range(2000):
        x = np.random.uniform(0.8, 4.2) 
        y = np.random.uniform(0.8, 4.2) 
        i = round((5 - y)/0.10)
        j = round(x/0.10)
        if(costmaps[num_map][(num_orientation*10)+10][i][j]>1.8):
            l_end.append(x)
            l_end.append(y)
    found = False

    for i in range(0,len(l_first),2):
        for j in range(0,len(l_end),2):
            d = sqrt((l_first[i]-l_end[j])**2+(l_first[i+1]-l_end[j+1])**2)
            if (d>=1.5 and d<1.8):
                x_middle = (l_first[i]+l_end[j])/2
                y_middle = (l_first[i+1]+l_end[j+1])/2
                i_middle = round((5 - y_middle)/0.10)
                j_middle = round(x_middle/0.10)
                if(costmaps[num_map][(num_orientation*10)+6][i_middle][j_middle]>2):
                    print("d", d)
                    print(i_middle,j_middle)
                    x_path.append(l_first[i])
                    x_path.append(l_end[j])
                    y_path.append(l_first[i+1])
                    y_path.append(l_end[j+1])
                    found = True
                if (found):
                    break
            if (found):
                break
        if (found):
            break

    theta_initial = atan2(y_path[1]-y_path[0],x_path[1]-x_path[0])+np.random.uniform(0, 0.3) 
    print("num_map = ", num_map , "num_orientaion = ", num_orientation) 
    print("x_ref_points = ", x_path)
    print("y_ref_points = ", y_path)
    print("theta_0 = ", theta_initial)

    return x_path, y_path, theta_initial
# -*- coding: utf-8 -*-
"""
@author: SzuHaoWu
"""
import numpy as np
from numpy.linalg import inv
import math
from map_utils import bresenham2D
#import matplotlib.pyplot as plts



def motion(imuyaw,encoder,imutime,time):
    
    m = np.size(time)
    w,x,y,theta,v,v_wheel = np.zeros((m,)),np.zeros((m,)),np.zeros((m,)),np.zeros((m,)),np.zeros((m,)),np.zeros((2,m))
    d_left = (encoder[0,:]+encoder[2,:])/2*0.0022
    d_right = (encoder[1,:]+encoder[3,:])/2*0.0022
    
    for i in range(1,np.size(encoder,1)):
        #compute time different
        t_diff = time[i] - time[i-1]
        
        #compute w
        closed_index = np.argmin(np.abs(imutime - time[i]))
        w[i] = imuyaw[closed_index]
        
        
        #compute velocity
        v_wheel[0,i] = d_left[i]/t_diff
        v_wheel[1,i] = d_right[i]/t_diff
        v[i] = (v_wheel[0,i]+v_wheel[1,i])/2
        
        #compute theta
        theta[i] = theta[i-1] + w[i]*t_diff
        
        #comput location
        sinc = math.sin(w[i]*t_diff/2)/(w[i]*t_diff/2)
        x[i] = x[i-1] + v[i]*t_diff*sinc*math.cos(theta[i]+w[i]*t_diff/2)
        y[i] = y[i-1] + v[i]*t_diff*sinc*math.sin(theta[i]+w[i]*t_diff/2)
        
    return x,y,theta
    
    #plt.scatter(x,y)
    #plt.show()
def getVandW(imuyaw,encoder,imutime,time):
    w,v,v_wheel = np.zeros((4956,)),np.zeros((4956,)),np.zeros((2,4956))
    d_left = (encoder[0,:]+encoder[2,:])/2*0.0022
    d_right = (encoder[1,:]+encoder[3,:])/2*0.0022
    
    for i in range(1,np.size(encoder,1)):
        #compute time different
        t_diff = time[i] - time[i-1]
        
        #compute w
        closed_index = np.argmin(np.abs(imutime - time[i]))
        w[i] = imuyaw[closed_index]
        
        
        #compute velocity
        v_wheel[0,i] = d_left[i]/t_diff
        v_wheel[1,i] = d_right[i]/t_diff
        v[i] = (v_wheel[0,i]+v_wheel[1,i])/2
        
    return v,w
    
def motion_diff(v,w,theta,t,t_diff):
  
    #compute time different        
    
    #comput location
    sinc = math.sin(w[t]*t_diff/2)/(w[t]*t_diff/2)
    dx = v[t]*t_diff*sinc*math.cos(theta+w[t]*t_diff/2)
    dy = v[t]*t_diff*sinc*math.sin(theta+w[t]*t_diff/2)
    dtheta = w[t]*t_diff    
    return dx,dy,dtheta
    
    
    
    
def uniformscandata(scan,lidartime,time):
    result = np.zeros((np.size(scan,0),np.size(time)))
    for i in range(0,np.size(time)):
        closed_index = np.argmin(np.abs(lidartime - time[i]))
        result[:,i] = scan[:,closed_index]
    
    return result
    
def softmax(x):
    e_x = np.exp(x-np.max(x))
    return e_x / e_x.sum()    



def mapping(lamda,originx,originy, scanx,scany):
    ratio = 4
    map2D = []
    for i in range(len(scanx)):
        map2D.append(bresenham2D(originx,originy,scanx[i],scany[i]))
        
    for k in map2D:
        [m,n] = np.shape(k)
        for j in range(n):
#            if k[0,j] <1601 and k[1,j] < 1601:
            if j != n-1:    
                lamda[int(k[0,j]),int(k[1,j])] = lamda[int(k[0,j]),int(k[1,j])] + math.log(1/ratio)
            else:
                lamda[int(k[0,j]),int(k[1,j])] = lamda[int(k[0,j]),int(k[1,j])] + math.log(ratio)

    return lamda
    
def WorldtoMap(x,y):
    MAP = {}
    MAP['res']   = 0.05 #meters
    MAP['xmin']  = -20  #meters
    MAP['ymin']  = -20
    MAP['xmax']  =  20
    MAP['ymax']  =  20 
    MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
    MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8

    xis = np.ceil((x - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    yis = np.ceil((y - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    return xis,yis

def BodytoWorld(xs,ys,theta):
    trans = [[math.cos(theta),-(math.sin(theta)),0],[math.sin(theta),math.cos(theta),0],[0,0,1]]
    result = np.dot(trans,np.transpose([xs,ys,0]))
    return result

def BodytoWorld1(input,theta):
    trans = [[math.cos(theta),-(math.sin(theta)),0],[math.sin(theta),math.cos(theta),0],[0,0,1]]
    result = np.dot(trans,input)
    return result

def uniformDistoRgb(dis_stamps,rgb_stamps):
    result = np.zeros_like(rgb_stamps)
    for i in range(np.size(rgb_stamps)):
        closed_index = np.argmin(np.abs(dis_stamps - rgb_stamps[i]))
        result[i] = closed_index
    return result

def uniformRGBtoEncoderIndex(rgb_time,encoder_stamps):
    closed_index = np.argmin(np.abs(encoder_stamps - rgb_time))
    return closed_index

def pitchRotation(deg,input):
    rot = [[math.cos(deg),0,math.sin(deg)],[0,1,0],[-math.sin(deg),0,math.cos(deg)]]
    return np.dot(rot,input)

def intrinsics(pixel,zo):
    K = [[585.05108211,0,242.94140713],[0,585.05108211,315.83800193],[0,0,1]]
    result = zo*np.dot(inv(K),pixel)
    return result
    


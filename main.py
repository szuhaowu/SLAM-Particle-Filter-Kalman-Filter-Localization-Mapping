# -*- coding: utf-8 -*-
"""
@author: SzuHaoWu
"""
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from map_utils import mapCorrelation
from util import motion,uniformscandata,mapping,WorldtoMap,BodytoWorld,BodytoWorld1,softmax,getVandW,motion_diff,uniformDistoRgb,uniformRGBtoEncoderIndex,intrinsics,pitchRotation
import math
from numpy.random import randn


class MapandFilter():
    def __init__(self,disp_stamps,encoder_counts, encoder_stamps,imu_angular_velocity,imu_stamps,lidar_angle_max,lidar_angle_min,lidar_range_max,lidar_range_min,lidar_ranges,lidar_stamps,rgb_stamps):
        self.disp_stamps = disp_stamps
        self.encoder_counts = encoder_counts
        self.encoder_stamps = encoder_stamps
        self.imu_angular_velocity = imu_angular_velocity
        self.imu_stamps = imu_stamps
        self.lidar_angle_max = lidar_angle_max
        self.lidar_angle_min = lidar_angle_min
        self.lidar_range_max = lidar_range_max
        self.lidar_range_min = lidar_range_min
        self.lidar_ranges = lidar_ranges
        self.lidar_stamps = lidar_stamps
        self.rgb_stamps = rgb_stamps
        self.angles = np.arange(-135,135.25,0.25)*np.pi/180.0
        self.lidaroriginx = 0.298
        self.lidaroriginy = 0
        self.frame_size = 1201
    def InputMotion(self):
        [x,y,theta] = motion(self.imu_angular_velocity[2,:],self.encoder_counts,self.imu_stamps,self.encoder_stamps)
        v,w = getVandW(self.imu_angular_velocity[2,:],self.encoder_counts,self.imu_stamps,self.encoder_stamps)
        
        return x,y,theta,v,w
    
    
    def DeadReckoning(self):
         newranges = self.lidar_ranges
         # take valid indices
         valid = np.logical_and((newranges[:,0] < 30),(newranges[:,0]> 0.1))
         newranges = newranges[valid]
         angles = self.angles[valid]
         newranges = uniformscandata(newranges,self.lidar_stamps,self.encoder_stamps)
         lamda_nofilter = np.zeros((self.frame_size,self.frame_size)) 
         x,y,theta,v,w = self.InputMotion()
         for i in range(np.size(newranges,1)):
             
             xs = newranges[:,i]*np.cos(angles)+self.lidaroriginx
             ys = newranges[:,i]*np.sin(angles)+self.lidaroriginy
             [xs,ys,z] = BodytoWorld(xs,ys,theta[i])
             lidarxis,lidaryis =  WorldtoMap(x[i]+0.298*math.cos(theta[i]),y[i]+0.298*math.sin(theta[i]))
             xis,yis =  WorldtoMap(xs+x[i],ys+y[i])
             lamda_nofilter = mapping(lamda_nofilter,lidarxis,lidaryis,xis,yis)
         
         return lamda_nofilter
     
    def ParticleFilter(self,particles_number):
        newranges = self.lidar_ranges
        # take valid indices
        valid = np.logical_and((newranges[:,0] < 30),(newranges[:,0]> 0.1))
        newranges = newranges[valid]
        angles = self.angles[valid]
        
        newranges = uniformscandata(newranges,self.lidar_stamps,self.encoder_stamps)
        lamda = np.zeros((self.frame_size,self.frame_size))
        walkmap = np.zeros((self.frame_size,self.frame_size))
        slammap = np.zeros((self.frame_size,self.frame_size))
        now_tra = np.zeros((3,np.size(newranges,1)))
        x,y,theta,v,w = self.InputMotion()
        
        #particle number
        N=particles_number
        particle = np.zeros((N,3))
        W = np.ones(N)/N
        res = 0.05
        dif = 2
        x_im = np.arange(-30,30)
        y_im = np.arange(-30,30)
        x_dif,y_dif = np.arange(-dif*res,dif*res,res),np.arange(-dif*res,dif*res,res)
        now = np.zeros((3,))
        noise1 = np.array([0.05, 0.05, 0.1*np.pi/180])
        noise2 = np.array([0.01, 0.01, 0.03*np.pi/180])
        cors = []
        
        for j in range(N):
            noises = np.random.randn(1,3)*noise1  
            particle[j,:] = [x[0],y[0],theta[0]]+noises
            
            #
        for i in range(1,np.size(newranges,1)): 
            for j in range(N):
                #        print('particle'+str(j))
                noises = np.random.randn(1,3)*noise2
                dx,dy,dtheta = motion_diff(v,w,particle[j,2],i,self.encoder_stamps[i]-self.encoder_stamps[i-1])
                particle[j,:] = [particle[j,0]+dx,particle[j,1]+dy,particle[j,2]+dtheta]+noises
                
            particle[:,2] %= 2*np.pi
            xs = newranges[:,i]*np.cos(angles)+self.lidaroriginx
            ys = newranges[:,i]*np.sin(angles)+self.lidaroriginy
            [xs,ys,z]= BodytoWorld(xs,ys,theta[i])
            cors = []
            temp = np.zeros_like(lamda)
            temp[lamda>0] = 1
            temp[lamda<0] = -1
            for j in range(N):    
                particle_cor_x, particle_cor_y = x_dif+particle[j,0], y_dif+particle[j,1]
                cor = mapCorrelation(temp,x_im,y_im,np.vstack((xs, ys)),particle_cor_x,particle_cor_y)
                cors.append(np.max(cor))
        
        
            cors = W * np.array(cors)
            W = softmax(cors)
            best = np.argmax(W)    
            now = particle[best,:].copy()
            now_tra[:,i] = [now[0],now[1],now[2]]
            now_map_x,now_map_y = WorldtoMap(now[0],now[1])
            best_lidar_map_x,best_lidar_map_y = WorldtoMap(now[0]+0.298*math.cos(now[2]),now[1]+0.298*math.sin(now[2]))
            
            xis,yis =  WorldtoMap(xs+x[i],ys+y[i])
            lamda = mapping(lamda,best_lidar_map_x,best_lidar_map_y,xis,yis)
            bodyxis,bodyyis = WorldtoMap(x[i],y[i])
            walkmap[bodyxis,bodyyis] = 1
            slammap[now_map_x,now_map_y] = 1
            
        return lamda,now_tra,walkmap,slammap
        
        
    def PlotMap(self,lamda,lamda_nofilter,now_tra,walkmap,slammap):
        lamda[lamda>0] = 1
        lamda[lamda<0] = -1
        lamda_nofilter[lamda_nofilter>0] = 1
        lamda_nofilter[lamda_nofilter<0] = -1
        walkmap[walkmap<1] = -1
        slammap[slammap<1] = -1
        
        fig = plt.figure()
        plt.imshow(lamda,cmap='gray')
        fig2 = plt.figure()
        plt.imshow(slammap)
        fig1 = plt.figure()
        plt.imshow(lamda_nofilter,cmap='hot')
        fig3 = plt.figure()
        plt.imshow(walkmap)
    
    def TextureMapping(self,pitchdeg,now_tra):
        color_MAP = np.zeros((self.frame_size,self.frame_size,3))
        dis_index = uniformDistoRgb(self.disp_stamps,self.rgb_stamps)
        for i in range(np.size(self.rgb_stamps)):
            rgb_data = plt.imread('C:/Users/SzuHaoWu/Desktop/UCSD/ECE276A/ECE276A_HW2/data/dataRGBD/RGB20/rgb20_{}.png'.format(i+1))
            dis_data = plt.imread('C:/Users/SzuHaoWu/Desktop/UCSD/ECE276A/ECE276A_HW2/data/dataRGBD/Disparity20/disparity20_{}.png'.format(int(dis_index[i]+1)))
            dd = (-0.00304*dis_data+3.31)
            depth = 1.03/dd
            rgbi = np.around((dd*-4.5*1750.46+19276.0)/585.051 + np.arange(0, dis_data.shape[0]).reshape([-1, 1])*526.37/585.051)
            rgbj = np.around(np.tile((np.arange(0,dis_data.shape[1]).reshape([1,-1])*526.37+16662.0)/585.051,(dis_data.shape[0],1)))
            rgbi_flat = rgbi.flatten()
            rgbj_flat = rgbj.flatten()
            depth_flat = depth.flatten()
            temp = np.zeros_like(rgbi_flat)+1
            pixel = (np.vstack((rgbi_flat,rgbj_flat,temp)))
            Observation = intrinsics(pixel*10,depth_flat)
            camera_rotation = pitchRotation(pitchdeg,Observation)
            camera_bodyframe = camera_rotation+np.array([[0.18],[0.005],[0.36]])
            index = uniformRGBtoEncoderIndex(self.rgb_stamps[i],self.encoder_stamps)
            pic_worldframe = BodytoWorld1(camera_bodyframe,now_tra[2,index])
            
            temp_index = np.where(pic_worldframe[2,:] < 3.2)
            index_i = rgbi_flat[temp_index].astype('int')
            index_j = rgbj_flat[temp_index].astype('int')
            x_rgb_MAP,y_rgb_MAP = WorldtoMap(pic_worldframe[0,temp_index]+now_tra[0,index],pic_worldframe[1,temp_index]+now_tra[1,index])
            color_MAP[x_rgb_MAP,y_rgb_MAP,:] = rgb_data[index_i,index_j,:]
        return color_MAP
        
    def PlotTexture(self,lamda,color_MAP):
        lamda[lamda>0] = 1
        lamda[lamda<0] = -1    
        img = np.zeros((self.frame_size,self.frame_size,3))
        for i in range(3):   
            img[:,:,i][lamda>0] = 1
            img[:,:,i][lamda==0] = 192/255
            img[:,:,i][lamda<0] = -1
            for j in range(color_MAP.shape[0]):
                for k in range(color_MAP.shape[1]):        
                    if color_MAP[j,k,i] > 0 and img[j,k,i] != 192/255:    
                        img[j,k,i] = color_MAP[j,k,i]
        plt.imshow(img)
    
    
    
    
if __name__ == '__main__':
    pitchdeg = 0.36
    
    result = MapandFilter(disp_stamps,encoder_counts, encoder_stamps,imu_angular_velocity,imu_stamps,lidar_angle_max,lidar_angle_min,lidar_range_max,lidar_range_min,lidar_ranges,lidar_stamps,rgb_stamps)
    x,y,theta,v,w = result.InputMotion()
    lamda_nofilter = result.DeadReckoning()
    lamda,now_tra,walkmap,slammap = result.ParticleFilter(100)
    result.PlotMap(lamda,lamda_nofilter,now_tra,walkmap,slammap)
    color_MAP = result.TextureMapping(pitchdeg,now_tra)
    result.PlotTexture(lamda,color_MAP)
    
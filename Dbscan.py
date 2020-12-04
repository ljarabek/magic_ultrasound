import cv2utils as cv2u
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from multi_slice_viewer import multi_slice_viewer

video_ = cv2u.array_from_video(
    file='/media/leon/TOSHIBA EXT/faks/Projekti/ultrazvok/26081958/20181220/26081958_20181220_MSK-_VIDEO_0003.AVI')
video = video_[:, 69:166, 220:435]
del video_
try:
    raw_flow = np.load("flownp.npy")
except:
    prev_frame = None
    output = [np.zeros_like(video[0])] # da je isto št sličic
    raw_flow = [np.zeros_like(video[0,:,:,:2])]
    for i, frame in tqdm(enumerate(video)):
        if prev_frame is None:
            prev_frame = frame
            continue

        flow = cv2u.optical_flow(frame, prev_frame, only_flow=True)

        flow_np = np.array(flow)
        raw_flow.append(flow_np)
        #print(flow_np.shape)
        prev_frame = frame
        flow_np_ = np.pad(flow_np, ((0,0), (0,0), (0,1)), mode = "constant", constant_values=((0,0), (0,0), (0,0)))
        #plt.imshow(flow_np[...,0])
        #plt.show()
        output.append(flow_np_)
        #if i>10: break


    output+= np.min(output)
    output /= np.max(output)
    output *= 255
    output = np.array(output, dtype=np.uint8)
    video = np.array(video, dtype=np.uint8)
    raw_flow = np.array(raw_flow)
    np.save("flownp.npy", raw_flow)

    write = np.concatenate([video,output], axis=1)
    cv2u.array_to_video(write, "flow.avi", to_u_=False)

raw_flow = np.array(raw_flow)# (1806, 97, 215, 2)
cluster_points = list()
prev_ = (-1,-1,-1)
current_point = list()
for i,index in tqdm(enumerate(np.ndindex(raw_flow.shape))):
    point_coordinates = index[:-1]

    if point_coordinates!=prev_:
        current_point = [x for x in index]
        current_point.append(raw_flow[index])
        prev_ = point_coordinates
    else:
        current_point.append(raw_flow[index])
        cluster_points.append(current_point)

import pickle

with open("points.pkl", "wb") as f:
    pickle.dump(cluster_points,f) # MemoryError









"""new_vid = list()
differences = list()
prev_frame = 0
laplaces_ = list()
for i, frame in tqdm(enumerate(video)):
    lol = cv2.GaussianBlur(frame, ksize=(3,3), sigmaX=15)
    lol = cv2.Laplacian(lol, ddepth=cv2.CV_8U, ksize=13)
    delta = np.abs(lol-prev_frame)
    differences.append(delta)#/2 + frame/2)



    prev_frame=lol
    laplaces_.append(lol)
    new_vid.append(cv2.GaussianBlur(frame, ksize=(3,3), sigmaX=15))
differences = np.array(differences)
#
#graph = differences[1:].sum(axis=(1,2,3))
#plt.plot(graph[:500])
#plt.show()

cv2u.array_to_video(new_vid, "blurlol_laplace.avi", to_u_=True)
cv2u.array_to_video(laplaces_, "laplace.avi", to_u_=True)"""

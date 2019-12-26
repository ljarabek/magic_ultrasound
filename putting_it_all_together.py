import cv2utils as cv2u
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from multi_slice_viewer import multi_slice_viewer

video_ = cv2u.array_from_video(file='C:/Users\Leon\Desktop\projekti/Ultrazvok2\ALOKA_US/26081958/20181220/26081958_20181220_MSK-_VIDEO_0003.AVI')
video = video_[:,69:166,220:435]
new_vid = list()
del video_
for id_frame, frame in enumerate(video):
    video[id_frame] = cv2.GaussianBlur(frame, ksize=(3,3), sigmaX=15)
#video=new_vid
optical_flow = cv2u.optical_flow_from_array(video, levels=3, winsize=5, poly_n=5) #flags="OPTFLOW_FARNEBACK_GAUSSIAN")
cv2u.array_to_video(optical_flow, "lol.avi")
f,h,w,c = optical_flow.shape
optical_flow = optical_flow[:,5:h-5,10:w-5,:]

points = []
i=0
vid = []

for frame_ in tqdm(optical_flow):
    nonzero = np.transpose(np.nonzero(frame_[:, :, 0]))
    if nonzero.shape[0] != 0:
        db = DBSCAN(eps=5, min_samples=40).fit(X=nonzero)  # 15, 30
        for d in nonzero[db.labels_ == 0]:
            print(i)
            x, y = d
            video[i,x+5,y+5] = [255, 0, 0]
        for d in nonzero[db.labels_ == 1]:
            x, y = d
            video[i, x+5, y+5] = [0, 255, 0]
    i+=1
cv2u.array_to_video(video)
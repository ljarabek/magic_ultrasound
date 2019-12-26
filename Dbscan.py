import cv2utils as cv2u
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from multi_slice_viewer import multi_slice_viewer

video_ = cv2u.array_from_video(file='C:/Users\Leon\Desktop\projekti/Ultrazvok2\ALOKA_US/26081958/20181220/26081958_20181220_MSK-_VIDEO_0003.AVI')
video = video_[:,69:166,220:435]
del video_
new_vid = list()
differences = list()
prev_frame = 0
for i, frame in tqdm(enumerate(video)):
    lol = cv2.GaussianBlur(frame, ksize=(3,3), sigmaX=15)
    lol = cv2.Laplacian(lol, ddepth=cv2.CV_8U, ksize=13)
    differences.append(np.abs(lol-prev_frame))#/2 + frame/2)
    prev_frame=lol
    new_vid.append(cv2.GaussianBlur(frame, ksize=(3,3), sigmaX=15))
differences = np.array(differences)
#
graph = differences[1:].sum(axis=(1,2,3))
plt.plot(graph[:500])
plt.show()

cv2u.array_to_video(new_vid, "blurlol_laplace.avi", to_u_=True)
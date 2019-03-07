import cv2
from scipy.misc import toimage
import matplotlib.pyplot as plt
vidcap = cv2.VideoCapture('C:/ultrasound_testing\ALOKA_US/26081958/20181220/26081958_20181220_MSK-_VIDEO_0003.AVI')
success,prev_image = vidcap.read()
prev_image=prev_image[80:205,245:425]
prev_image =prev_image
count = 0

import numpy as np
from scipy import signal
from cv2utils import optical_flow, to_u




mn =5e5
mx=-5e5
images_us = []
images_transforms = []
while success:
    #cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
    success,image = vidcap.read()  # image shape 392 640 3
    image=image[80:205,245:425]
    image = np.array(image)

    count+=1
    print(count)
    u = optical_flow(image,prev_image)

    images_transforms.append(u)
    if np.max(u)> mx:
        mx= np.max(u)
    if np.min(u)<mn:
        mn=np.min(u)
    std = 2*np.std(u)
    indices = u<(np.mean(u)+std)
    u[indices]=0

    images_us.append(image)
    prev_image = image
    if count>1000:
        break

video = []
datapoints = []
#mn = np.min(images_transforms)
#mx = np.max(images_transforms)
for id, i in enumerate(images_transforms):
    datapoints.append(np.sum(i))
    i-=mn
    i/= mx
    i*=255
    i=np.uint8(i)
    frame = np.concatenate((images_us[id],i))
    video.append(frame)
    print(frame.shape)
    #plt.imshow(frame)
    #plt.show()
#plt.plot(datapoints)
#plt.show()


out = cv2.VideoWriter('../lotltest.avi',apiPreference=0, fourcc= cv2.VideoWriter_fourcc(*'DIVX'), fps=18, frameSize=(video[0].shape[1],video[0].shape[0]))

for imag in video:
    out.write(imag)
#out.release()

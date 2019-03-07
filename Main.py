import cv2
from scipy.misc import toimage
vidcap = cv2.VideoCapture('C:/ultrasound_testing\ALOKA_US/05061951/20181220/05061951_20181220_MSK-_VIDEO_0002.AVI')
success,prev_image = vidcap.read()
prev_image =prev_image[130:280,214:470]
count = 0
import matplotlib.pyplot as plt


import numpy as np
from scipy import signal

def optical_flow(one, two):
    """
    method taken from (https://chatbotslife.com/autonomous-vehicle-speed-estimation-from-dashboard-cam-ca96c24120e4)
    """
    one_g = cv2.cvtColor(one, cv2.COLOR_RGB2GRAY)
    two_g = cv2.cvtColor(two, cv2.COLOR_RGB2GRAY)
    hsv = np.zeros(np.array(prev_image).shape) #    hsv = np.zeros((392,640, 3))

    # set saturation
    hsv[:,:,1] = cv2.cvtColor(two, cv2.COLOR_RGB2HSV)[:,:,1]
    # obtain dense optical flow paramters
    flow = cv2.calcOpticalFlowFarneback(one_g, two_g, flow=None,    #https://www.programcreek.com/python/example/89313/cv2.calcOpticalFlowFarneback
                                        pyr_scale=0.5, levels=1, winsize=2, #15 je bil winsize
                                        iterations=2,
                                        poly_n=5, poly_sigma=1.1, flags=0)
    # convert from cartesian to polar
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # hue corresponds to direction
    hsv[:,:,0] = ang * (180/ np.pi / 2)
    # value corresponds to magnitude
    hsv[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    # convert HSV to int32's
    hsv = np.asarray(hsv, dtype= np.float32)
    rgb_flow = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    return rgb_flow

def to_u(a):
    a -= np.min(a)
    a /= np.max(a)
    #print('Read a new frame: ', success)
    a *= 255
    # u = np.concatenate((u, np.uint8(image)), axis=0)
    return np.uint8(a)

#def get_optical_flow(a,b):

mn =5e5
mx=-5e5
images_us = []
images_transforms = []
while success:
    #cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
    success,image = vidcap.read()  # image shaoe 392 640 3
    image = np.array(image[130:280,214:470])
    #plt.imshow(image)
    #plt.show()
    #u=[]
    count+=1
    print(count)
    u = optical_flow(image,prev_image)



    # TO ZA RAZLIKE
    #u = np.float32(image)-np.float32(prev_image)
    #u -= np.min(u)
    #u/=np.max(u)
    #print('Read a new frame: ', success)

    # TO ZA DVF
    #u = optical_flow(image,prev_image)
    #u=to_u(u)


    images_transforms.append(u)
    if np.max(u)> mx:
        mx= np.max(u)
    if np.min(u)<mn:
        mn=np.min(u)
    std = 2*np.std(u)
    indices = u<(np.mean(u)+std)
    u[indices]=0

    images_us.append(image)
    """DELA
    image = np.array(image)
    u=[]
    count+=1
    print(count)
    u = optical_flow(image,prev_image)

    print('Read a new frame: ', success)
    u=to_u(u)

    images.append(u)
    """
        #u=toimage(u)
        #cv2.convertTo(u, CV_)
        #plt.imshow(image)
        #plt.show()
        #plt.imshow(prev_image)
        #plt.show()
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
plt.plot(datapoints)
plt.show()


out = cv2.VideoWriter('lotltest22.avi',apiPreference=0, fourcc= cv2.VideoWriter_fourcc(*'DIVX'), fps=18, frameSize=(video[0].shape[1],video[0].shape[0]))

for imag in video:
    out.write(imag)
#out.release()

from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
import cv2utils
import cv2
frames_ = np.array(np.load('videotest.npy'))

## extract DVF and trim edges:

min_samples = 80
eps = 15
f,h,w,c = np.array(frames_).shape
frames = frames_[:,int(h/2+10):int(h-10),10:w-10,:]
ultrasound = np.array(frames_[:,0:frames_.shape[1]//2, :],dtype=np.uint8)
#plt.imshow(frames[96,:,:])
#plt.show()
#frame = frames[96,:,:]
vid = []
points=[]
i=0
for frame_ in frames:
    print(i)
    i+=1
    #if i==2:
    #    break
    #print(np.array(frame).shape)
    #print(indices[0,:])
    nonzero = np.transpose(np.nonzero(frame_[:,:,0]))
    if nonzero.shape[0]!=0:
        db  = DBSCAN(eps=eps,min_samples=min_samples).fit(X=nonzero) #15, 30
        #print(db.labels_)
        #print("lol")
        frame =  np.array(frame_)#, np.float32)
        for d in nonzero[db.labels_==0]:
            x,y = d
            frame_[x,y,:] = [255,0,0]
            points.append([i,x,y])
        for d in nonzero[db.labels_ == 1]:
            x, y = d
            frame_[x, y, :] = [0, 255, 0]
    #if frame.dtype==np.float32:
    #    vid.append(cv2utils.to_u(frame))
    #else:
    frame_ = cv2.resize(frame_, dsize=(ultrasound.shape[2], ultrasound.shape[1]))
    vid.append(frame_)
    #print(np.shape(frame_))
    #plt.imshow(frame_)
    #plt.show()


#vid = np.array(np.tile(np.expand_dims(vid,-1), (3)), dtype=np.uint8)
#vid = np.array(vid, dtype=np.uint8)
#vid = np.array(vid)
#ultrasound[vid.any()>50]= [0, 255, 0]
#ultrasound[vid==]= [255, 255, 0]

#vid = np.ma.masked_where(vid!=[255,0,0],vid)
#vid = np.array((ultrasound+vid)/2, dtype=np.uint8)
for i,x,y in points:
    ultrasound[i,x,y] = [0,0,255]
    print(i,x,y)
#print(vid.shape)
cv2utils.array_to_video(ultrasound,file="letsdothis_%s_%s.avi"%(min_samples,eps))



#for index, label in zip(nonzero,db.labels_):
#    x,y = index
#    if label == -1:
#        continue
#    else:
#
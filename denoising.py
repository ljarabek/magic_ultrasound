import cv2
from scipy.misc import toimage
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from cv2utils import optical_flow, to_u, generator_from_video, array_from_video, array_to_video, to_u
import tensorflow as tf
from tqdm import tqdm
#tf.enable_eager_execution()



def denoised_array(video_array = array_from_video(), winsize = 30):
    print(np.array(video_array).shape)
    for frame in video_array:
        print(frame.shape)
    array = []
    win = int(winsize/2)
    for frame_no in tqdm(range(100)):#len(video_array))):
        print("lol " + str(frame_no))
        try:
            v_window = video_array[frame_no-win:frame_no+win]
            frame = video_array[frame_no] - np.average(v_window, axis=0)
            frame /= np.std(v_window)
            frame += np.abs(np.min(frame))
            frame*=10
            array.append(frame)#to_u(frame))
        except:
            array.append(to_u(video_array[frame_no]))
    return np.array(array, dtype=np.uint8)


array_to_video(denoised_array())





#data = tf.data.Dataset().batch(5).from_generator(generator_from_video, output_types=tf.float32, output_shapes=(392,640,3))
"""data = tf.data.Dataset().batch(5).from_generator(generator_from_video, output_types=tf.float32, output_shapes=(392,640,3))
iter = data.make_initializable_iterator()
ex = iter.get_next()

#392 640 3
hidden_dim = np.int32(35)
h,w,c = np.int32(392), np.int32(640), np.int32(3)
x = tf.layers.flatten(ex)
encode = {'weights': tf.Variable(tf.truncated_normal(
            [h*w*c, hidden_dim], dtype=tf.float32)),
            'biases': tf.Variable(tf.truncated_normal([hidden_dim],
                                                      dtype=tf.float32))}
decode = {'biases': tf.Variable(tf.truncated_normal([h*w*c], dtype=tf.float32)),'weights': tf.Variable(tf.truncated_normal(
            [hidden_dim, h*w*c], dtype=tf.float32))}

encoded = tf.nn.tanh(tf.matmul(x, encode['weights']) + encode['biases'])

decoded = tf.matmul(encoded, decode['weights'])+decode['biases']


loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(x,decoded))))     #recon loss (u can use crossentropy loss = -tf.reduce_mean(x_ * tf.log(decoded)))

train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
with tf.Session() as sess:
    #m = tf.matmul(data, 4.8)
    sess.run(iter.initializer)
    sess.run(tf.global_variables_initializer())
    for i in range(5000):
        print(np.array(sess.run(data)).shape)
        #_, los = sess.run((train, loss))
        #print(str(los))
"""

"""mn =5e5
mx=-5e5
images_us = []
images_transforms = []
while success:
    #cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
    success,image = vidcap.read()  # image shape 392 640 3
    image=image[80:205,245:425]
    image = np.array(image)
    plt.imshow(prev_image)
    plt.show()



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



out = cv2.VideoWriter('../denoise_test.avi',apiPreference=0, fourcc= cv2.VideoWriter_fourcc(*'DIVX'), fps=18, frameSize=(video[0].shape[1],video[0].shape[0]))

for imag in video:
    out.write(imag)
#out.release()"""

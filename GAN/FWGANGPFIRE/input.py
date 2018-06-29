import os
import tensorflow as tf
from PIL import Image  # 注意Image,后面会用到
import matplotlib.pyplot as plt
import numpy as np
from utils import *

cwd = 'E:/FWGANGPFIRE/'
classes = {'train'}  # 人为 设定 2 类
writer = tf.python_io.TFRecordWriter("eyes.tfrecords")  # 要生成的文件

for index, name in enumerate(classes):
    class_path = cwd + name + '\\'
    for img_name in os.listdir(class_path):
        img_path = class_path + img_name  # 每一个图片的地址

        img = Image.open(img_path)
       # img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img = img.resize((256, 256))
        img_raw = img.tobytes()  # 将图片转化为二进制格式
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))  # example对象对label和image数据进行封装
        writer.write(example.SerializeToString())  # 序列化为字符串

writer.close()

# def read_and_decode(filename):  # 读入dog_train.tfrecords
#     #sess = tf.Session()
#
#     filename_queue = tf.train.string_input_producer([filename])  # 生成一个queue队列
#     #tf.train.start_queue_runners(sess=sess)
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
#     features = tf.parse_single_example(serialized_example,
#                                        features={
#                                          #  'label': tf.FixedLenFeature([], tf.int64),
#                                            'img_raw': tf.FixedLenFeature([], tf.string),
#                                        })  # 将image数据和label取出来
#
#     img = tf.decode_raw(features['img_raw'], tf.uint8)
#     img = tf.reshape(img, [256, 256, 3])  # reshape为128*128的3通道图片
#     img = (tf.cast(img, tf.float32) -127.5)/127.5  # 在流中抛出img张量
#    # img -= np.mean(img, axis=0)  # zero-center
#   # img /= np.std(img, axis=0)  # normalize
#     #img = tf.cast(img, tf.float32) / 255.0  # 在流中抛出img张量
#    # label = tf.cast(features['label'], tf.int32)  # 在流中抛出label张量
#     images = tf.train.shuffle_batch([img],
#                                           batch_size=256, capacity=2000,
#                                           min_after_dequeue=1000)
#    # images  =images .eval(session = sess)
#     return images
# data_X = read_and_decode("fire.tfrecords")
# sess = tf.Session()
# for idx in range(0, 16):
#     threads = tf.train.start_queue_runners(sess=sess, coord=tf.train.Coordinator())
#     batch_images = sess.run(data_X[idx * 16:(idx + 1) * 16])
#     tot_num_samples = min(16, 64)
#     image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))
#     image_dims = [256, 256, 3]
#     """ random condition, random noise """
#     inputs = tf.placeholder(tf.float32, [16] + image_dims, name='real_images')
#    # z_sample = (np.random.uniform(-1, 1, size=(16, 100)) * 127.5) + 127.5
#     samples =sess.run(inputs, feed_dict={inputs:batch_images})
#
#     save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
#                 check_folder(
#                    'E:/FWGANGPFIRE/real') + '/'  + '_epoch%03d' % idx + 'real.png')


'''
filename_queue = tf.train.string_input_producer(["fire.tfrecords"]) #读入流中
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)   #返回文件名和文件
features = tf.parse_single_example(serialized_example,
                                   features={
                                       'label': tf.FixedLenFeature([], tf.int64),
                                       'img_raw' : tf.FixedLenFeature([], tf.string),
                                   })  #取出包含image和label的feature对象
image = tf.decode_raw(features['img_raw'], tf.uint8)
image = tf.reshape(image, [64, 64, 3])
label = tf.cast(features['label'], tf.int32)
with tf.Session() as sess: #开始一个会话
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)
    for i in range(256):
        example, l = sess.run([image,label])#在会话中取出image和label
        img=Image.fromarray(example, 'RGB')#这里Image是之前提到的
        img.save(cwd+str(i)+'_''Label_'+str(l)+'.jpg')#存下图片
        print(example, l)
    coord.request_stop()
    coord.join(threads)
'''
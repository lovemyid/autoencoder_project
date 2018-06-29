from __future__ import division
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import os, gzip
import _pickle
import tensorflow as tf
import tensorflow.contrib.slim as slim

def read_and_decode(filename):  # 读入dog_train.tfrecords
    #sess = tf.Session()

    filename_queue = tf.train.string_input_producer([filename])  # 生成一个queue队列
    #tf.train.start_queue_runners(sess=sess)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                         #  'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })  # 将image数据和label取出来

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [256, 256, 3])  # reshape为128*128的3通道图片
    img = tf.cast(img, tf.float32) /255.0  # 在流中抛出img张量
   # img -= np.mean(img, axis=0)  # zero-center
  # img /= np.std(img, axis=0)  # normalize
    #img = tf.cast(img, tf.float32) / 255.0  # 在流中抛出img张量
   # label = tf.cast(features['label'], tf.int32)  # 在流中抛出label张量
    images = tf.train.shuffle_batch([img],
                                          batch_size=256, capacity=2000,
                                          min_after_dequeue=1000)
   # images  =images .eval(session = sess)
    return images

def unpickle(relpath):
    print(relpath)
    with open(relpath, 'rb') as fp:
        d = _pickle.load(fp,encoding='bytes')
    return d
def prepare_input(data=None, labels=None):
    image_height = 256
    image_width = 256
    image_depth = 3
    assert(data.shape[1] == image_height * image_width * image_depth)
    assert(data.shape[0] == labels.shape[0])
    #do mean normaization across all samples
    mu = np.mean(data, axis=0)
    mu = mu.reshape(1,-1)
    sigma = np.std(data, axis=0)
    sigma = sigma.reshape(1, -1)
    data = data - mu
    data = data / sigma
    is_nan = np.isnan(data)
    is_inf = np.isinf(data)
    if np.any(is_nan) or np.any(is_inf):
        print('data is not well-formed : is_nan {n}, is_inf: {i}'.format(n= np.any(is_nan), i=np.any(is_inf)))
    data = data.reshape([-1,image_depth, image_height, image_width])
    data = data.transpose([0, 2, 3, 1])
    data = data.astype(np.float32)
    return data, labels

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def get_image(image_path, input_height, input_width, resize_height=64, resize_width=64, crop=True, grayscale=False):
    image = imread(image_path, grayscale)
    return transform(image, input_height, input_width, resize_height, resize_width, crop)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, grayscale = False):
    if (grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)

def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width, resize_height=64, resize_width=64, crop=True):
    if crop:
        cropped_image = center_crop(image, input_height, input_width, resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.

def save_scattered_image(z, id, z_range_x, z_range_y, name='scattered_image.jpg'):
    N = 10
    plt.figure(figsize=(8, 6))
    plt.scatter(z[:, 0], z[:, 1], c=np.argmax(id, 1), marker='o', edgecolor='none', cmap=discrete_cmap(N, 'jet'))
    plt.colorbar(ticks=range(N))
    axes = plt.gca()
    axes.set_xlim([-z_range_x, z_range_x])
    axes.set_ylim([-z_range_y, z_range_y])
    plt.grid(True)
    plt.savefig(name)

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)
from __future__ import division
import numpy as np
import os
import tensorflow as tf

try:
    import urllib2
except ImportError:
    import urllib.request as urllib

from datasets import imagenet
from nets import resnet_v1
from preprocessing import vgg_preprocessing

from tensorflow.contrib import slim

from datasets import dataset_utils
import h5py
from tqdm import tqdm
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans, MiniBatchKMeans
import scipy.misc
import math
from shutil import copyfile

def maybe_mkdirs(path):
    if type(path) is list:
        for path in path:
            maybe_mkdirs(path)
    else:
        if not os.path.exists(path):
            os.makedirs(path)


url = "http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz"
checkpoints_dir = '/tmp/checkpoints'
samples_dir = '/scratch/sagea/temp'
maybe_mkdirs([checkpoints_dir, samples_dir])

dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)

end_point_name = 'global_pool'
image_size = 224
# proj_dim = 0
pca_dim = 128
batch_size = 64
n_clusters = 128
algo = 'mkmeans'
label_name = 'labels/resnet/%s_%i_%i' % (end_point_name, pca_dim, n_clusters)
print(label_name)
dataset = '/home/sagea/scratch/data/cifar-10/cifar_data.hdf5'

if not os.path.exists(dataset):
    maybe_mkdirs(os.path.dirname(dataset))
    source = ['', 'home'] + [dataset.split('/')[2]] + ['scratch'] +dataset.split('/')[3:]
    copyfile('/'.join(source), dataset)
    print('copied data')
hdf_file = h5py.File(dataset, 'r+')
data = hdf_file['data']
_feats = []

with tf.Graph().as_default():
    images = tf.placeholder(tf.int32, (batch_size, 32, 32, 3))
    processed_images = tf.map_fn(lambda img: vgg_preprocessing.preprocess_image(img, image_size, image_size,
                                                                               is_training=False), images, dtype=tf.float32)
    #processed_images = tf.expand_dims(processed_image, 0)

    # Create the model, use the default arg scope to configure the batch norm parameters.
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        # 1000 classes instead of 1001.
        _, end_points = resnet_v1.resnet_v1_50(processed_images, num_classes=1000, is_training=False)

    for key in end_points.iterkeys():
        print(key)

    end_point_used = end_points[end_point_name]
    features = tf.reshape(end_point_used, [batch_size, -1])

    # random projection
    features_len = features.get_shape().as_list()[1]
    # rand_proj = tf.random_normal((features_len, proj_dim), stddev=1/features_len, seed=123456)
    # rand_proj_exp = tf.tile(tf.expand_dims(rand_proj, 0), [batch_size, 1, 1])
    # features_rp = tf.matmul(features, rand_proj)

    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'resnet_v1_50.ckpt'),
        slim.get_model_variables('resnet_v1_50'))

    # samples_file = h5py.File(os.path.join(samples_dir, 'temp_samples.hdf5'), 'w')
    # samples_data = samples_file.create_dataset('data', shape=((len(data) // batch_size)*batch_size, features_len))
    with tf.Session() as sess:
        init_fn(sess)
        for idx in tqdm(xrange(len(data) // batch_size)):
            _imgs = data[idx*batch_size:(idx+1)*batch_size]
            # samples_data[idx*batch_size:(idx+1)*batch_size] = (sess.run(features, feed_dict={images: _imgs.transpose((0, 2, 3, 1))}))
            _feats.append(sess.run(features, feed_dict={images: _imgs.transpose((0, 2, 3, 1))}))
    _feats = np.concatenate(_feats)

    pca = PCA(n_components=pca_dim)
    # pca = TruncatedSVD(n_components=pca_dim)
    feats_pca = pca.fit_transform(_feats)

    if algo == 'kmeans':
        print('Starting KMeans algorithm...')
        k_means = KMeans(n_clusters=n_clusters, max_iter=10000, n_jobs=-1)
        try:
            labels = k_means.fit_predict(feats_pca)
        except MemoryError:
            algo = 'mkmeans'
    if algo == 'mkmeans':
        print('Starting MiniBatchKMeans algorithm...')
        k_means = MiniBatchKMeans(n_clusters=n_clusters, max_iter=10000, verbose=1, max_no_improvement=10000,
                                  batch_size=1000, tol=0.0, n_init=5)
        labels = k_means.fit_predict(feats_pca)

    c_probs = np.bincount(labels) / float(len(labels))
    hdf_lbl = hdf_file.create_dataset(label_name, data=labels)
    hdf_lbl.attrs['probs'] = c_probs

# check clusters
def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def imshow(images, size):
    return scipy.misc.imshow(merge(images, size))
target_path = '/home/sagea/scratch/cluster_check/cifar/%s' % label_name
maybe_mkdirs(target_path)
n_images = len(data)
print('Images: %i' % n_images)
n_labels = len(labels)
n_images = min(n_images, n_labels)
print('Labels: %i' % n_labels)
clusters = [[] for i in range(n_clusters)]
for idx in tqdm(range(n_images)):
    clusters[labels[idx]].append(data[idx])
for n, cluster in tqdm(enumerate(clusters)):
    clus = np.array(cluster).transpose((0,2,3,1))
    imsave(clus, (int(math.ceil(math.sqrt(len(cluster)))), int(math.ceil(math.sqrt(len(cluster))))), os.path.join(target_path, 'cluster_%02d.png' % n))

hdf_file.close()

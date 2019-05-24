# -*- coding:utf-8 -*-
import numpy as np
import scipy.misc
import time
import h5py

def make_generator(hdf5_file, n_images, batch_size, res, label_name=None): # label_name == label path
    epoch_count = [1]
    def get_epoch():
        # print('new epoch!')
        images = np.zeros((batch_size, 3, res, res), dtype='int32') # return images' shape
        images_HH_temp = np.zeros((batch_size,3,16,16), dtype='int32') # 3*16*16 = 768

        labels = np.zeros(batch_size, dtype='int32')
        indices = range(n_images)
        random_state = np.random.RandomState(epoch_count[0])
        random_state.shuffle(indices)
        epoch_count[0] += 1
        for n, i in enumerate(indices):
            # assuming (B)CWH format
            images[n % batch_size] = hdf5_file['data/data'][i]
            images_HH_temp[n % batch_size] = hdf5_file['data/data_HH'][i]
            if label_name is not None:
                labels[n % batch_size] = hdf5_file[label_name][i]
            if n > 0 and n % batch_size == 0:
                images =  np.array(images)
                labels = np.array(labels) 
                images_HH = np.array(images_HH_temp).reshape(batch_size,-1)
                assert (images.shape == (batch_size,3,32,32))
                assert (labels.shape == (batch_size,))
                # print('images_HH shape is:',images_HH.shape)
                assert (images_HH.shape == (batch_size,768))
                yield (images, labels,images_HH) # return numpy type data; images_HH as noise; 返回的noise形状是(bs,768)
    return get_epoch


def load(batch_size, data_file='/home/maolongchun/logo-syns/wgan/data/cifar-new.hdf5', resolution=32, label_name=None):
    hdf5_file = h5py.File(data_file, 'r')
    n_images = len(hdf5_file['data/data'])
    if label_name is not None:
        n_labels = len(hdf5_file[label_name])
        print('Labels: %i' % n_labels)
        n_images = min(n_images, n_labels)
    print('Images: %i' % n_images)
    return make_generator(hdf5_file, n_images, batch_size, res=resolution, label_name=label_name)


def load_new(cfg):
    label_name = cfg.LABELS if cfg.LABELS != 'None' else None # cfg.LABELS is a path to labels in hdf5 file.
    return load(cfg.BATCH_SIZE, cfg.DATA, cfg.OUTPUT_RES, label_name=label_name)


if __name__ == '__main__':
    train_gen, valid_gen = load(64)
    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        print "{}\t{}".format(str(time.time() - t0), batch[0][0,0,0,0])
        if i == 1000:
            break
        t0 = time.time()
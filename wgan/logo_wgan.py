# -*- coding:utf-8 -*-
"""Improved Wasserstein GAN code for the CVPR paper
   "Logo Synthesis and Manipulation with Clustered Generative Adversarial Networks"
   Based in large parts on the official codebase for "Improved Training of Wasserstein GANs" by Gulrajani et al.
   Available at https://github.com/igul222/improved_wgan_training"""

import os
import argparse

import tflib as lib
import tflib.save_images
import tflib.cifar10
import tflib.twitter_images
import tflib.hdf5_images
import tflib.small_imagenet
import tflib.inception_score
import tflib.architectures

import h5py
import numpy as np
import tensorflow as tf
import json
from shutil import copyfile

import time
import locale

locale.setlocale(locale.LC_ALL, '')


class Config(object):
    """Config class used to set all the parameter used for GAN training and inference.
       It can be used by passing an instantiated object of the script to the training function.
       Alternatively, all parameters can aso be passed as named parameters on the command line."""
    
    def __init__(self, initial_data=None, **kwargs):     # 这个配置是为好几个模型准备的，所以显得有些乱，注意区分不同模型的配置，一种模型往往只会用到这里面的一小部分。
        # default values sorted by aphabet
        self.ACGAN = 0  # BOOLEAN If CONDITIONAL, whether to use ACGAN or "vanilla" conditioning
        self.ACGAN_SCALE = 1.  # How to scale the critic's ACGAN loss relative to WGAN loss
        self.ACGAN_SCALE_G = 0.1  # How to scale generator's ACGAN loss relative to WGAN loss
        self.ARCHITECTURE = 'resnet-32' #'resnet-32'  # used GAN architecture 在这里更改使用的模型结构 ###########################

        self.BATCH_SIZE = 64  # Critic batch size
        self.bn_init = True
        self.CONDITIONAL = 0  # BOOLEAN Whether to train a conditional or unconditional model

        self.DIM_G = 128  # Generator dimensionality
        self.DIM_D = 128  # Critic dimensionality
        self.DECAY = 1  # BOOLEAN Whether to decay LR over learning
        self.DATA_LOADER = 'hdf5'  # Data format to be used                      ##################################
        self.DATA = 'data/LLD-icon-sharp.hdf5'  # Path to training data (folder or file depending on fromat)       ##############################

        self.GEN_BS_MULTIPLE = 2  # Generator batch size, as a multiple of BATCH_SIZE 作为批量大小的倍数

        self.ITERS = 100000  # How many iterations to train for
        self.INCEPTION_FREQUENCY = 10000 # How frequently to calculate Inception score
        self.KEEP_CHECKPOINTS = 5  # Number of checkpoints to keep (long-term, spread out over entire training time)

        self.LABELS = 'labels/resnet1/rc_128'  # Path to labels: Either the filesystem location of a pickle file containing the labels or the path to the label dataset within a HDF5 file  ####################
        self.LAYER_COND = 1  # BOOLEAN feed the labels to every layer in generator and discriminator
        self.LR = 0  # Initial learning rate [0 --> default]
        self.LAMBDA = 10  # gradient penalty lambda

        self.MODE = 'wgan-gp' #'wgan-gp'  # training mode            #####################
        
        self.N_GPUS = 1
        self.NORMALIZATION_G = 1  # BOOLEAN Use batchnorm in generator?
        self.NORMALIZATION_D = 0  # BOOLEAN Use batchnorm (or layernorm) in critic?
        self.N_LABELS = 128  # Number of label classes :label 类别的数目             ################################
        self.N_CRITIC = 5  # Critic steps per generator steps (except for lsgan and DCGAN training modes)
        self.N_GENERATOR = 3  # Generator steps per critic step for DCGAN training mode

        self.OUTPUT_RES = 32 # icon图标的宽高为32x32，也是Res-Generator最后生成图像的大小
        self.OUTPUT_DIM = self.OUTPUT_RES * self.OUTPUT_RES * 3  # 一张图片的总像素数
        self.RUN_NAME = 'sharp_128_Z-LC'  # name for this experiment run          #################################
        self.SUMMARY_FREQUENCY = 1  # How frequently to write out a tensorboard summary

        if self.N_GPUS not in [1, 2]:
            raise Exception('Only 1 or 2 GPUs supported!')
        if len(self.DATA) == 0:
            raise Exception('Please specify path to data directory in gan_cifar.py!')

        # 不同模型使用不同的学习率：
        if self.LR == 0:
            if self.MODE == 'wgan-gp':
                if self.ARCHITECTURE == 'resnet-32':
                    self.LR = 2e-4
                else:
                    self.LR = 1e-4
            if self.MODE == 'wgan':
                self.LR = 5e-5
            if self.MODE == 'dcgan':
                self.LR = 2e-4
            if self.MODE == 'lsgan':
                self.LR = 1e-4
            if self.MODE == 'ae-l2':
                self.LR = 2e-4

        # read config from dict or keywords，从字典读取配置并设置新的属性，这部分只是起补充作用，从initial_data或kwargs中补充新的配置或更改原有的属性值。
        if initial_data is not None:
            for key in initial_data:
                setattr(self, key, initial_data[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])


    def __str__(self):
        return str(self.__dict__)

    def __cmp__(self, other):
        return self.__dict__ == other.__dict__


class WGAN(object):
    """This class contains all functions used for GAN training and inference.
       An object can be instantiated by loading the parameters from file, passing them as a dict or as named parameters.
       session:     TF session
       load_config: Path to the config.json file containing the parameters (typically of a previously trained GAN)
       config_dict: Dictionary representation a Config object
       **kwargs:    Additional named parameters (will override those loaded from file or dict)"""

    def __init__(self, session, load_config=None, config_dict=None, **kwargs):
        self.session = session
        self.sampler_initialized = 0
        self.current_iter = 0
        self.update_moving_stats = False
        self.t_train = tf.placeholder(tf.bool)
        # create config object
        if load_config is not None:
            with open(os.path.join('runs', load_config, 'config.json'), 'r') as f:
                loaded_dict = json.load(f)
            cfg = Config(loaded_dict, **kwargs) 
        else:
            cfg = Config(config_dict, **kwargs)
        self.cfg = cfg

        if cfg.CONDITIONAL and (not cfg.ACGAN) and (not cfg.LAYER_COND) and (not cfg.NORMALIZATION_D):
            print "WARNING! Conditional model without normalization in D might be effectively unconditional!"

        # returns a (Generator, Discriminator) pair according to config
        def get_architecture(cfg):
            if cfg.ARCHITECTURE == 'dcgan-32':
                return tflib.architectures.Generator_DCGAN_32, tflib.architectures.Discriminator_DCGAN_32
            if cfg.ARCHITECTURE == 'dcgan-64':
                return tflib.architectures.Generator_DCGAN_64, tflib.architectures.Discriminator_DCGAN_64
            if cfg.ARCHITECTURE == 'm-dcgan-64':
                return tflib.architectures.Generator_MultiplicativeDCGAN_64, \
                       tflib.architectures.Discriminator_MultiplicativeDCGAN_64
            if cfg.ARCHITECTURE == 'resnet-32':
                return tflib.architectures.Generator_Resnet_32, tflib.architectures.Discriminator_Resnet_32
            if cfg.ARCHITECTURE == 'resnet-64':
                return tflib.architectures.Generator_Resnet_64, tflib.architectures.Discriminator_Resnet_64
            if cfg.ARCHITECTURE == 'resnet-128':
                return tflib.architectures.Generator_Resnet_128, tflib.architectures.Discriminator_Resnet_128
            if cfg.ARCHITECTURE == 'b-resnet-64': # 带batch_normalization的ResNetv2 结构，64类
                return tflib.architectures.Generator_Bottleneck_Resnet_64, \
                       tflib.architectures.Discriminator_Bottleneck_Resnet_64
            if cfg.ARCHITECTURE == 'fc-64':
                return tflib.architectures.Generator_FC_64, tflib.architectures.Discriminator_FC_64

        self.Generator, self.Discriminator = get_architecture(cfg)

        lib.print_model_settings_dict(cfg.__dict__)

        self.run_dir = os.path.join('runs', cfg.RUN_NAME) # ./runs/wgan_run_dir/
        self.save_dir = os.path.join(self.run_dir, 'checkpoints') # ./runs/wgan_run_dir/checkpoints ,note,this is a dir not file.
        # self.ground_truth_sample_dir = os.path.join(self.run_dir,'ground_truth') # 只存储对应的真实图片
        self.sample_dir = os.path.join(self.run_dir, 'samples')
        self.cluster_0 = os.path.join(self.sample_dir,'cluster0')
        self.cluster_1 = os.path.join(self.sample_dir,'cluster1')
        self.cluster_2 = os.path.join(self.sample_dir,'cluster2')
        self.cluster_3 = os.path.join(self.sample_dir,'cluster3')
        self.sample_path_list = [self.cluster_0,self.cluster_1,self.cluster_2,self.cluster_3]
        self.tb_dir = os.path.join(self.run_dir, 'tensorboard')
        self.saver = None

        def maybe_mkdirs(path):
            # path是一个路径字符串或者路径列表
            # 如果是一个路径列表集合，则每个建立一个路径/目录；如果不是列表则直接根据路径建立一个目录；这个函数设置了循环调用，很巧妙。
            if type(path) is list:
                for path in path:
                    maybe_mkdirs(path)
            else:
                if not os.path.exists(path):
                    os.makedirs(path)

        maybe_mkdirs([self.run_dir, self.save_dir, self.sample_dir, self.tb_dir,self.sample_path_list])

        # 输入数据：
        self._iteration = tf.placeholder(tf.int32, shape=None)
        self.all_real_data_int = tf.placeholder(tf.int32, shape=(cfg.BATCH_SIZE, 3, cfg.OUTPUT_RES, cfg.OUTPUT_RES))
        self.all_real_labels = tf.placeholder(tf.int32, shape=(cfg.BATCH_SIZE,))
        self.new_noise = tf.placeholder(tf.float32,shape=[cfg.BATCH_SIZE,128]) # 定义待传入的高频噪声的形状
        
        # self.noise = tf.concat([np.random.normal(size=(cfg.BATCH_SIZE,128)).astype('float32'),self.new_noise],axis=1) # (bs,256)

        # self.fixed_noise = tf.constant(np.random.normal(size=(100, 256)).astype('float32')) # 固定的高斯噪声，(100,128)维,用于检验G的产出
        if cfg.LAYER_COND:
            self.fixed_labels = tf.one_hot(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10, dtype='int32'),depth=cfg.N_LABELS) # depth定义了one-hot标签的维度，即128类为128维。（100,128）维的固定one-hot向量
        else:
            self.fixed_labels = tf.constant(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10, dtype='int32'))
        self.z = None

        self.y = None
        self.label_probs = None
        # if cfg.DATA_LOADER in ['hdf5', 'twitter'] and cfg.LABELS != 'None':
        #     with h5py.File(cfg.DATA, 'r') as f:
        #         cfg.label_probs = f[cfg.LABELS].attrs['probs']

        # copy data to scratch, only works for single files!
        if (cfg.DATA_LOADER == 'hdf5' or cfg.DATA_LOADER == 'twitter') and not os.path.exists(self.cfg.DATA):
            if not os.path.exists(os.path.dirname(self.cfg.DATA)):
                os.makedirs(os.path.dirname(self.cfg.DATA))
            source = ['', 'home'] + [self.cfg.DATA.split('/')[2]] + ['scratch'] + self.cfg.DATA.split('/')[3:]
            copyfile('/'.join(source), self.cfg.DATA) # 从绝对路径复制文件到目标路径cfg.DATA
            print('copied data')

        def data_sample():
            cluster = []
            label_sample = []
            with h5py.File('data/LLD-icon-sharp2.hdf5','r') as f:
                data = np.array(f['data'])
                HH_noise = np.array(f['data_hh'])
                label = np.array(f['labels/resnet1/rc_128'])
                class_indices = [np.where(label == i)[0] for i in range(self.cfg.N_LABELS)] #######################################  range(N_LABELS)
                sample_cluster = [16,45,103,124]
                for cluster_label in sample_cluster:
                    index_list = np.random.choice(class_indices[cluster_label],12)
                    sample_images = data[index_list] # (12,32,32,3)
                    lib.save_images.save_images(sample_images,'sample_{}.png'.format(cluster_label))   

                    HH_noise_sample = HH_noise[index_list]
                    label_ = label[index_list]
                    cluster.append(HH_noise_sample)
                    label_sample.append(label_)
            return data,HH_noise,label,cluster,label_sample # (bs,3,32,32),(bs,128),(4,12,3,32,32),(4,12,)

        self.data,self.HH_noise,self.label,cluster,label_sample = data_sample()
        assert(len(self.data)==len(self.HH_noise)==len(self.label))
        self.cluster0, self.label0 = cluster[0], tf.one_hot(np.array(label_sample[0],dtype='int32'),depth=self.cfg.N_LABELS)
        self.cluster1, self.label1 = cluster[1], tf.one_hot(np.array(label_sample[1],dtype='int32'),depth=self.cfg.N_LABELS)
        self.cluster2, self.label2 = cluster[2], tf.one_hot(np.array(label_sample[2],dtype='int32'),depth=self.cfg.N_LABELS)
        self.cluster3, self.label3 = cluster[3], tf.one_hot(np.array(label_sample[3],dtype='int32'),depth=self.cfg.N_LABELS)
        # ______________________________________________________________________________ # 构造函数__init__到此结束

    def get_data_loader(self):
        if self.cfg.DATA_LOADER == 'pickle':
            return lib.batchloader
        if self.cfg.DATA_LOADER == 'cifar-10':
            return lib.cifar10
        if self.cfg.DATA_LOADER == 'lld-logo':
            return lib.twitter_images
        if self.cfg.DATA_LOADER == 'hdf5':
            return lib.hdf5_images
        if self.cfg.DATA_LOADER == 'imagenet-small':
            return lib.small_imagenet

    # restores a GAN model from checkpoint
    def restore_model(self):
        # initialize saver
        if self.saver is None:
            self.saver = tf.train.Saver()
        # try to restore checkpoint
        ckpt = tf.train.get_checkpoint_state(self.save_dir)
        if ckpt:
            with open(os.path.join(self.run_dir, 'config.json'), 'r') as f:
                old_dict = json.load(f)
            new_dict = self.cfg.__dict__
            equal = True
            for key, value in old_dict.iteritems():
                if (key != 'train') and (key[:3] != 'bn_') and new_dict[key] != value:
                    print('New: %s: %s' % (key, new_dict[key]))
                    print('Old: %s: %s' % (key, value))
                    equal = False
            if not equal:
                raise Exception('Config for existing checkpoint is not the same, aborting!')
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
            self.current_iter = int(np.loadtxt(os.path.join(self.save_dir, 'last_iteration')))
            print('model restored.')
        else:
            with open(os.path.join(self.run_dir, 'config.json'), 'w') as f:
                json.dump(self.cfg.__dict__, f) # 把字典存到文件confi.json中去

    # initializes the sampler (used for GAN inference) 用于G的推理
    def _init_sampler(self):
        t_False = tf.constant(True, dtype=tf.bool)
        self.update_moving_stats = False
        y_shape = (None, self.cfg.N_LABELS) if self.cfg.LAYER_COND else (None,) # 如果使用条件gan模型，则等于label种类数。
        self.y = tf.placeholder((tf.float32 if self.cfg.LAYER_COND else tf.int32), shape=y_shape, name='y') # y标签即用于G也用于D

        self.z = tf.placeholder(tf.float32, shape=(None, 128), name='z') # z的shape和label有多少种类有关
        all_real_data = tf.reshape(2 * ((tf.cast(self.all_real_data_int, tf.float32) / 256.) - .5),
                                   [self.cfg.BATCH_SIZE, self.cfg.OUTPUT_DIM]) # 把输入图片从4D整到2D

        all_real_data += tf.random_uniform(shape=[self.cfg.BATCH_SIZE, self.cfg.OUTPUT_DIM], minval=0.,
                                           maxval=1. / 128)  # dequantize 一个batch的2D real data 加上相同形状的均匀分布噪声
        self.sampler = self.Generator(self.cfg, n_samples=12, labels=self.y, noise=self.z, is_training=self.t_train)
        # if sampler_d not needed, is_training parameter can be removed
        self.sampler_d = self.Discriminator(self.cfg, inputs=all_real_data, labels=self.y, is_training=self.t_train)
        self.restore_model()
        self.sampler_initialized = True

    # returns a batch of latent space samples by Generator 产生一个个cluster的图片
    def sample_g(self,cluster_idx, z=None, y=None):
        # input: cluster_idx,z:HH_noise,y:labels.
        if z is None:
            z = np.random.normal(size=(12, 128)).astype('float32') #如果z为None则传入高斯噪声，否则传入高频噪声
        if y is None:
            y = np.array([cluster_idx] * 12, dtype='int32') # 每个cluster_idx 代表一个cluster的标签；这里取都属于同一标签的12个样本
        if self.cfg.LAYER_COND and len(y.shape) == 1:
            y = tf.one_hot(y,depth=self.cfg.N_LABELS)
        if not self.sampler_initialized:
            self._init_sampler()
        samples = self.session.run(self.sampler, feed_dict={self.z: z, self.y: y, self.t_train: False})
        samples = ((samples + 1.) * (255. / 2)).astype('int32')
        samples = samples.reshape((len(z), 3, self.cfg.OUTPUT_RES, self.cfg.OUTPUT_RES))
        return samples.transpose((0, 2, 3, 1)) # (bs,32,32,3)

    def sample_d(self, input, y):  ########### 这个函数也没用上 #########
        if self.cfg.LAYER_COND and len(y.shape)==1 : # 逻辑运算符的优先级仅高于lambda，低于比较运算符的优先级
            y = np.eye(self.cfg.N_LABELS)[y]
        if not self.sampler_initialized:
            self._init_sampler()
        return self.session.run(self.sampler_d, feed_dict={self.all_real_data_int: input,
                                                           self.y: y, self.t_train: False})

    def z_sampler(self, size=None): ############ 这个函数没用上 ########## 返回(BATCH_SIZE,128)维的高斯噪声
        if size is None:
            size = self.cfg.BATCH_SIZE
        return np.random.normal(size=(size, 128)).astype('float32')



    def train(self):
        cfg = self.cfg
        t_True = tf.constant(True, dtype=tf.bool)
        t_False = tf.constant(False, dtype=tf.bool)
        DEVICES = ['/gpu:{}'.format(i) for i in xrange(cfg.N_GPUS)] # N_GPUS = 1 deafult, DEVICES = ['/gpu:0']
        if len(DEVICES) == 1:  # Hack because the code assumes 2 GPUs 强行搞成两个GPU
            DEVICES = [DEVICES[0], DEVICES[0]] # DEVICES = ['/gpu:0','/gpu:0'] ，len(DEVICES) = 2

        if cfg.LAYER_COND:                       # self.all_real_labels是占位变量，大小为batch_size
            labels_splits = tf.split(tf.one_hot(self.all_real_labels, depth=cfg.N_LABELS), len(DEVICES), axis=0) # (2,32,32),两份，每份32个，每个是深度为32的one-hot标签
        else:
            labels_splits = tf.split(self.all_real_labels, len(DEVICES), axis=0)
        
        if cfg.LAYER_COND:
            noise_splits = tf.split(self.new_noise,len(DEVICES),axis=0) 

        # G生成的样本叫fake_data_splits，数目是半个batch_size大小。相应的labels_splits是由传入的真实label分成两等分，与之对应。
        fake_data_splits = []
        for i, device in enumerate(DEVICES):
            with tf.device(device):

                # G的默认输入是在其结构定义中的；label这里用的已经是真是标签；我的目的是在其结构定义的地方把默认输入换成与标签对应的高频通道噪声。
                # 传入的label_splits[i] shape:(bs/2,cfg.N_LABELS),eg. (32,128),one-hot.
                fake_data_splits.append(self.Generator(cfg, cfg.BATCH_SIZE / len(DEVICES), labels_splits[i],HH_noise=noise_splits[i] ,is_training=self.t_train))  

        # 这一步对传入的真实数据的处理就看不懂了，转换数据类型再除以256.属于归一化，再减0.5、乘以2是什么意思？
        # 归一化把数据归一到[0,1),减去0.5变成[-0.5,0.5),再乘以2再次放缩到[-1,1)范围内。 紧接着就是再加上一个均匀分布进一步反量化。
        all_real_data = tf.reshape(2 * ((tf.cast(self.all_real_data_int, tf.float32) / 256.) - .5),
                                   [cfg.BATCH_SIZE, cfg.OUTPUT_DIM])
        all_real_data += tf.random_uniform(shape=[cfg.BATCH_SIZE, cfg.OUTPUT_DIM], minval=0.,
                                           maxval=1. / 128)  # dequantize 为什么所有的real_data都要加一个反量化的矩阵？？？

        all_real_data_splits = tf.split(all_real_data, len(DEVICES), axis=0) # 把数据平均分成两个部分，每部分只有半个batchsize大小了。

        DEVICES_B = DEVICES[:len(DEVICES) / 2] # DEVICES[0] = '/gpu:0'
        DEVICES_A = DEVICES[len(DEVICES) / 2:] # DEVICES[1] = '/gpu:0'

        disc_costs = [] # 计算D上的损失
        disc_acgan_costs = []
        disc_acgan_accs = []
        disc_acgan_fake_accs = []
        # 把一个batchsize大小的真实数据和G生成的fakedata拼接在一起；传入的标签label_splits拼接在一起； 然后统一传给D计算损失
        # 这样拼接后的大小是 2个batchsize大小
        for i, device in enumerate(DEVICES_A):
            with tf.device(device): # device_a = '/gpu:0'
                real_and_fake_data = tf.concat([
                    all_real_data_splits[i], # i = 0
                    all_real_data_splits[len(DEVICES_A) + i],
                    fake_data_splits[i],
                    fake_data_splits[len(DEVICES_A) + i]], axis=0)

                real_and_fake_labels = tf.concat([
                    labels_splits[i],
                    labels_splits[len(DEVICES_A) + i],
                    labels_splits[i],
                    labels_splits[len(DEVICES_A) + i]], axis=0)

                disc_all, disc_all_acgan = self.Discriminator(cfg, real_and_fake_data, real_and_fake_labels) #把真实数据和生成数据放在一起，用同样的标签，送给D鉴别。

                disc_real = disc_all[:cfg.BATCH_SIZE / len(DEVICES_A)] # D产生损失的前半段是真实数据的损失，后半段是fake数据的损失
                disc_fake = disc_all[cfg.BATCH_SIZE / len(DEVICES_A):]
                # 不同模型不同计算损失的方式：
                if cfg.MODE == 'wgan' or cfg.MODE == 'wgan-gp':
                    disc_costs.append(tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)) # wgan-gp模型计算两个分布之间的损失
                elif cfg.MODE == 'dcgan':
                    disc_costs.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real,labels=tf.ones_like(disc_real))) / 2.) # dcgan只计算D在real上的交叉熵
                elif cfg.MODE == 'lsgan':
                    disc_costs.append(tf.reduce_mean((disc_real - 1) ** 2) / 2.)

                # ACGAN cost, if applicable
                if cfg.CONDITIONAL and cfg.ACGAN:
                    disc_acgan_costs.append(tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=disc_all_acgan[:cfg.BATCH_SIZE / len(DEVICES_A)],
                            labels=real_and_fake_labels[:cfg.BATCH_SIZE / len(DEVICES_A)])
                    ))
                    disc_acgan_accs.append(tf.reduce_mean(
                        tf.cast(
                            tf.equal(
                                tf.to_int32(tf.argmax(disc_all_acgan[:cfg.BATCH_SIZE / len(DEVICES_A)], dimension=1)),
                                real_and_fake_labels[:cfg.BATCH_SIZE / len(DEVICES_A)]
                            ),
                            tf.float32
                        )
                    ))
                    disc_acgan_fake_accs.append(tf.reduce_mean(
                        tf.cast(
                            tf.equal(
                                tf.to_int32(tf.argmax(disc_all_acgan[cfg.BATCH_SIZE / len(DEVICES_A):], dimension=1)),
                                real_and_fake_labels[cfg.BATCH_SIZE / len(DEVICES_A):]
                            ),
                            tf.float32
                        )
                    ))
        
        # device A 从D计算损失；device B从G生成的数据和真实数据之间的差异计算损失，且这种区别仅在于wgan上，在其他模型上没有变化。
        for i, device in enumerate(DEVICES_B):
            with tf.device(device):
                real_data = tf.concat([all_real_data_splits[i], all_real_data_splits[len(DEVICES_A) + i]], axis=0) # shape:(BATCH_SIZE,OUTPUT_RES)，即(64,3*32*32)
                fake_data = tf.concat([fake_data_splits[i], fake_data_splits[len(DEVICES_A) + i]], axis=0) # shape:(64,3*32*32)
                labels = tf.concat([
                    labels_splits[i],
                    labels_splits[len(DEVICES_A) + i],
                ], axis=0)
                if cfg.MODE == 'wgan-gp':
                    alpha = tf.random_uniform(
                        shape=[cfg.BATCH_SIZE / len(DEVICES_A), 1],
                        minval=0.,
                        maxval=1.
                    )
                    differences = fake_data - real_data
                    interpolates = real_data + (alpha * differences) ############ 插值 ###########
                    # 相比于device A ，B 中D的输入变了，变成了插值；A中D的输入是单纯把real和fake拼接起来作为输入；B中把D的输出再和输入做一次梯度计算。
                    gradients = tf.gradients(self.Discriminator(cfg, interpolates, labels)[0], [interpolates])[0] # ys中每个y对xs中每个x求导之和为一个tensor，最后输出的tensor个数和xs中含有的x数目一致
                    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
                    gradient_penalty = tf.reduce_mean((slopes - 1.0)**2)
                    disc_costs.append(cfg.LAMBDA * gradient_penalty) # cfg.LAMBDA = 10

                elif cfg.MODE == 'dcgan':
                    disc_costs.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                                                             labels=tf.zeros_like(
                                                                                                 disc_fake))) / 2.)
                elif cfg.MODE == 'lsgan':
                    disc_costs.append(tf.reduce_mean((disc_fake - 0) ** 2) / 2.)

        disc_wgan = tf.add_n(disc_costs) / len(DEVICES_A) # D的wgan损失，即两个数据分部之间的损失，也称“推土机距离”；计算损失的时候要除以2是因为损失算是在相同数据上算了两遍，尤其是对于非wgan模型
        tf.summary.scalar('disc_cost', disc_wgan)

        if cfg.CONDITIONAL and cfg.ACGAN:
            disc_acgan = tf.add_n(disc_acgan_costs) / len(DEVICES_A)
            tf.summary.scalar('disc_acgan', disc_acgan)
            disc_acgan_acc = tf.add_n(disc_acgan_accs) / len(DEVICES_A)
            tf.summary.scalar('disc_acgan_acc', disc_acgan_acc)
            disc_acgan_fake_acc = tf.add_n(disc_acgan_fake_accs) / len(DEVICES_A)
            tf.summary.scalar('disc_acgan_fake_acc', disc_acgan_fake_acc)
            disc_cost = disc_wgan + (cfg.ACGAN_SCALE * disc_acgan)
        else:
            disc_cost = disc_wgan
        tf.summary.scalar('disc_cost', disc_cost)

        # ---- Generator costs ---- #
        gen_costs = []
        gen_acgan_costs = []
        for i, device in enumerate(DEVICES):
            with tf.device(device):
                n_samples = cfg.GEN_BS_MULTIPLE * cfg.BATCH_SIZE / len(DEVICES) # 2*64/2=64
                fake_labels = tf.concat([labels_splits[(i + n) % len(DEVICES)] for n in range(cfg.GEN_BS_MULTIPLE)],axis=0)

                disc_fake, disc_fake_acgan = self.Discriminator(cfg, self.Generator(cfg, n_samples, fake_labels,HH_noise=self.new_noise ,is_training=self.t_train),fake_labels) # D鉴别G的损失

                if cfg.MODE == 'wgan' or cfg.MODE == 'wgan-gp':
                    gen_costs.append(-tf.reduce_mean(disc_fake))


                elif cfg.MODE == 'dcgan':
                    gen_costs.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                                                            labels=tf.ones_like(
                                                                                                disc_fake))))
                elif cfg.MODE == 'lsgan':
                    gen_costs.append(tf.reduce_mean((disc_fake - 1) ** 2))
                # ACGAN cost, if applicable
                if disc_fake_acgan is not None:
                    gen_acgan_costs.append(tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_fake_acgan, labels=fake_labels)
                    ))
        gen_cost = (tf.add_n(gen_costs) / len(DEVICES))
        if cfg.CONDITIONAL and cfg.ACGAN:
            gen_cost += (cfg.ACGAN_SCALE_G * (tf.add_n(gen_acgan_costs) / len(DEVICES)))
        gen_costs = gen_costs

        # ---- Optimizer functions ---- #

        if cfg.DECAY:  #默认为True               # self._iteration占位符变量，待传入
            decay = tf.maximum(0., 1. - (tf.cast(self._iteration, tf.float32) / cfg.ITERS)) # 10万次迭代
        else:
            decay = 1.

        # 不同模型使用不同的优化器，而且G和D还要分别指定各自的优化器
        if cfg.MODE == 'wgan-gp':
            gen_opt = tf.train.AdamOptimizer(learning_rate=cfg.LR * decay, beta1=0., beta2=0.9)
            disc_opt = tf.train.AdamOptimizer(learning_rate=cfg.LR * decay, beta1=0., beta2=0.9)
        elif cfg.MODE == 'dcgan':
            gen_opt = tf.train.AdamOptimizer(learning_rate=cfg.LR * decay, beta1=0.5)
            disc_opt = tf.train.AdamOptimizer(learning_rate=cfg.LR * decay, beta1=0.5)
        elif cfg.MODE == 'wgan' or cfg.MODE == 'lsgan':
            gen_opt = tf.train.RMSPropOptimizer(learning_rate=cfg.LR * decay)
            disc_opt = tf.train.RMSPropOptimizer(learning_rate=cfg.LR * decay)
        gen_gv = gen_opt.compute_gradients(gen_cost, var_list=lib.params_with_name('Generator')) # 通过G损失计算G的优化梯度；
        disc_gv = disc_opt.compute_gradients(disc_cost, var_list=lib.params_with_name('Discriminator.')) # 通过D损失计算D的优化梯度
        # add BN dependencies
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            gen_train_op = gen_opt.apply_gradients(gen_gv)
            disc_train_op = disc_opt.apply_gradients(disc_gv)

        if cfg.MODE == 'wgan': # wgan有个裁切操作来满足变量约束
            clip_ops = []
            for var in lib.params_with_name('Discriminator'): #通过名字获取变量
                clip_bounds = [-.01, .01]
                clip_ops.append(tf.assign(var, tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])))
            clip_disc_weights = tf.group(*clip_ops)



        def save_images(cluster,label_sample,log_dir_list,frame):
            for idx,log_dir in enumerate(log_dir_list):
                HH_noise_ = cluster[idx] #(12,128)
                label_sample_ = label_sample[idx] # (bs,)
                label_sample_ = tf.one_hot(label_sample_,depth=cfg.N_LABELS)
                fixed_samples = self.Generator(cfg, 12, labels=label_sample_, HH_noise=HH_noise_) # 用于产生可视图像的G；传入的label和噪声要有对应关系才行
                samples = self.session.run(fixed_samples)
                # Function to generate samples
                def generate_image(log_dir, frame): # 把由G产生的假样本保存保存到指定路径为图片
    
                    samples = ((samples + 1.) * (255. / 2)).astype('int32')   # 为什么要做这一步处理？我看很多地方都有这种处理
                    lib.save_images.save_images(samples.reshape((12, 3, cfg.OUTPUT_RES, cfg.OUTPUT_RES)),
                                                os.path.join(log_dir, 'G_samples_{}_{}.png'.format(idx,frame)))
                generate_image(log_dir,frame)


        # Function for generating samples
        # todo: check possibility to change this to none
        # fixed_noise_samples = self.Generator(cfg, 100, self.fixed_labels, noise=self.fixed_noise, is_training=self.t_train) # 用于产生可视图像的G；传入的label和噪声要有对应关系才行
        fixed_noise_samples0 = self.Generator(cfg, 12, self.label0, HH_noise=self.cluster0, is_training=self.t_train) 
        fixed_noise_samples1 = self.Generator(cfg, 12, self.label1, HH_noise=self.cluster1, is_training=self.t_train) 
        fixed_noise_samples2 = self.Generator(cfg, 12, self.label2, HH_noise=self.cluster2, is_training=self.t_train) 
        fixed_noise_samples3 = self.Generator(cfg, 12, self.label3, HH_noise=self.cluster3, is_training=self.t_train) 

        # Function to generate samples
        def generate_image0(log_dir, frame): # 把由G产生的假样本保存保存到指定路径为图片
            samples = self.session.run(fixed_noise_samples0, feed_dict={self.t_train: True})
            samples = ((samples + 1.) * (255. / 2)).astype('int32')   # 为什么要做这一步处理？我看很多地方都有这种处理
            lib.save_images.save_images(samples.reshape((12, 3, cfg.OUTPUT_RES, cfg.OUTPUT_RES)),
                                        os.path.join(log_dir, 'samples_{}.png'.format(frame)))

        def generate_image1(log_dir, frame): # 把由G产生的假样本保存保存到指定路径为图片
            samples = self.session.run(fixed_noise_samples1, feed_dict={self.t_train: True})
            samples = ((samples + 1.) * (255. / 2)).astype('int32')   # 为什么要做这一步处理？我看很多地方都有这种处理
            lib.save_images.save_images(samples.reshape((12, 3, cfg.OUTPUT_RES, cfg.OUTPUT_RES)),
                                        os.path.join(log_dir, 'samples_{}.png'.format(frame)))

        def generate_image2(log_dir, frame): # 把由G产生的假样本保存保存到指定路径为图片
            samples = self.session.run(fixed_noise_samples2, feed_dict={self.t_train: True})
            samples = ((samples + 1.) * (255. / 2)).astype('int32')   # 为什么要做这一步处理？我看很多地方都有这种处理
            lib.save_images.save_images(samples.reshape((12, 3, cfg.OUTPUT_RES, cfg.OUTPUT_RES)),
                                        os.path.join(log_dir, 'samples_{}.png'.format(frame)))

        def generate_image3(log_dir, frame): # 把由G产生的假样本保存保存到指定路径为图片
            samples = self.session.run(fixed_noise_samples3, feed_dict={self.t_train: True})
            samples = ((samples + 1.) * (255. / 2)).astype('int32')   # 为什么要做这一步处理？我看很多地方都有这种处理
            lib.save_images.save_images(samples.reshape((12, 3, cfg.OUTPUT_RES, cfg.OUTPUT_RES)),
                                        os.path.join(log_dir, 'samples_{}.png'.format(frame)))

        # Function for calculating inception score
        if self.label_probs is not None:
            elems = tf.convert_to_tensor(range(cfg.N_LABELS))
            samples = tf.multinomial(tf.log([self.label_probs]), 100)  # note log-prob
            fake_labels_100 = elems[tf.cast(samples[0], tf.int32)]
        else:
            fake_labels_100 = tf.cast(tf.random_uniform([100]) * cfg.N_LABELS, tf.int32) # 从均匀分布产生100个假标签
        if cfg.LAYER_COND:
            fake_labels_100 = tf.cast(tf.one_hot(fake_labels_100, cfg.N_LABELS), tf.float32)
        
        # sample_inception_loader = self.get_data_loader()
        # gen_samples = sample_inception_loader.load_new(cfg) # 又创建一个新的生成器对象
        # def sample_loader():
        #     while True:
        #         for img, lab, img_hh in gen_samples():
        #             yield img, lab, img_hh
        # sample_gen = sample_loader()
        # _ , lab, img_hh = sample_gen.next() # 生成器都自带next方法，去下一个batch的数据
        # fake_lab = tf.cast(tf.one_hot(lab,depth= cfg.N_LABELS),tf.float32)

 

        def batch_generator(data,HH_noise,label,batch_size): # generator function return a generator object.
            
            batch_num = len(data)/batch_size
            for batch in range(batch_num-1):
                dt = data[batch*batch_size:(batch+1)*batch_size]
                lab = label[batch:(batch+1)*batch_size]
                hh_noise = HH_noise[batch:(batch+1)*batch_size]
                yield dt,lab,hh_noise

        # inception_sample_loader = self.get_data_loader()
        # sample_gen = inception_sample_loader.load_new(cfg)
        # def inf_gen():
        #     while True:
        #         for batch_data,batch_label,batch_noise in sample_gen():
        #             yield batch_data,batch_label,batch_noise

        # label = tf.one_hot(tf.cast(label,tf.int32),depth=cfg.N_LABELS)
        incep_gen = batch_generator(self.data,self.HH_noise,self.label,cfg.BATCH_SIZE)
        batch_data,batch_label,batch_noise = incep_gen.next() # 如果是生成器对象该对象可以直接调用.next()方法
        batch_label = tf.one_hot(batch_label,depth=cfg.N_LABELS)
        samples_64 = self.Generator(cfg, 64, batch_label, HH_noise=batch_noise, is_training=self.t_train) # 用于计算inception_score的G

        def get_inception_score(n): # n=100
            all_samples = []
            for i in xrange(n / 100): # xrange产生的是一个生成器对象
                # todo
                all_samples.append(session.run(samples_64, feed_dict={self.t_train: True}))
            all_samples = np.concatenate(all_samples, axis=0)
            all_samples = ((all_samples + 1.) * (255.99 / 2)).astype('int32') # (all_samples+1)*127
            all_samples = all_samples.reshape((-1, 3, cfg.OUTPUT_RES, cfg.OUTPUT_RES)).transpose(0, 2, 3, 1) #3通道32x32的图片
            return lib.inception_score.get_inception_score(list(all_samples))


        for name, grads_and_vars in [('G', gen_gv), ('D', disc_gv)]: # 分别输出G和D的参数名称和参数
            print "{} Params:".format(name)
            total_param_count = 0
            for g, v in grads_and_vars:
                shape = v.get_shape()
                shape_str = ",".join([str(x) for x in v.get_shape()])

                param_count = 1
                for dim in shape:
                    param_count *= int(dim)
                total_param_count += param_count

                if g == None:
                    print "\t{} ({}) [no grad!]".format(v.name, shape_str)
                else:
                    print "\t{} ({})".format(v.name, shape_str)
            print "Total param count: {}".format(
                locale.format("%d", total_param_count, grouping=True))


        run_number = len(next(os.walk(self.tb_dir))[1]) + 1
        summaries_merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(os.path.join(self.tb_dir, 'run_%i' % run_number), session.graph)
        # separate dev disc cost
        dev_cost_summary = tf.summary.scalar('dev_disc_cost', disc_cost)
        session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.restore_model()
        
        # for train
        data_loader = self.get_data_loader()
        train_gen = data_loader.load_new(cfg) # train_gen是load_new(cfg)函数返回的一个函数对象，不是生成器对象
        def inf_train_gen():  # 从数据集加载数据和label
            while True:
                for images, _labels, images_HH in train_gen():
                    yield images, _labels, images_HH # images_HH for G inputs.

        gen = inf_train_gen()
        sample_images, sample_labels, sample_images_HH = gen.next() # shape:(bs,3,32,32),(bs,),(bs,368), BCHW 
        # if sample_labels is None:
        #     sample_labels = [0] * cfg.BATCH_SIZE
        # # Save a batch of ground-truth samples 
        # _x_r = self.session.run(real_data, feed_dict={self.all_real_data_int: sample_images}) # 传入(bs,3,32,32)真实图片
        # _x_r = ((_x_r + 1.) * (255.99 / 2)).astype('int32')
        # lib.save_images.save_images(_x_r.reshape((cfg.BATCH_SIZE, 3, cfg.OUTPUT_RES, cfg.OUTPUT_RES)),
        #                             os.path.join(self.run_dir, 'samples_groundtruth.png'))


        if cfg.CONDITIONAL and cfg.ACGAN:
            _costs = {'cost': [], 'wgan': [], 'acgan': [], 'acgan_acc': [], 'acgan_fake_acc': []}
        else:
            _costs = {'cost': []}


        # 下面这一段代码定义G和D的交互训练策略，比如训练三次G训练一次D或者训练一次G训练5次D，选择那个策略看用的模型
        for iteration in xrange(self.current_iter, cfg.ITERS):
            start_time = time.time()
            if cfg.MODE == 'dcgan':
                gen_iters = cfg.N_GENERATOR # 3，如果是DCGAN模型则训练三次G训练一次D
            else:
                gen_iters = 1 # 训练1次
            if iteration > 0 and '_labels' in locals() and '_data_HH' in locals():
                for i in xrange(gen_iters):
                    _ = self.session.run([gen_train_op], feed_dict={self._iteration: iteration,
                                                                    self.all_real_labels: _labels,
                                                                    self.new_noise:_data_HH,
                                                                    self.t_train: True})

            if (cfg.MODE == 'dcgan') or (cfg.MODE == 'lsgan'):
                disc_iters = 1 
            else:
                disc_iters = cfg.N_CRITIC # 不是dcgan和lsgan，每训练一次G训练五次D

            for i in xrange(disc_iters):

                _data, _labels,_data_HH = gen.next() # 训练D有两个三个输入：来自G和real的数据以及相同的label.

                if _labels is None:
                    _labels = [0] * cfg.BATCH_SIZE
                if cfg.CONDITIONAL and cfg.ACGAN:
                    _summary, _disc_cost, _disc_wgan, _disc_acgan, _disc_acgan_acc, _disc_acgan_fake_acc, _ = \
                        self.session.run([summaries_merged, disc_cost, disc_wgan, disc_acgan, disc_acgan_acc,
                                          disc_acgan_fake_acc, disc_train_op],
                                         feed_dict={self.all_real_data_int: _data, 
                                                    self.all_real_labels: _labels,
                                                    self.new_noise: _data_HH,
                                                    self._iteration: iteration, 
                                                    self.t_train: True})
                    _costs['cost'].append(_disc_cost)
                    _costs['wgan'].append(_disc_wgan)
                    _costs['acgan'].append(_disc_acgan)
                    _costs['acgan_acc'].append(_disc_acgan_acc)
                    _costs['acgan_fake_acc'].append(_disc_acgan_fake_acc)
                else:
                    _summary, _disc_cost, _ = self.session.run([summaries_merged, disc_cost, disc_train_op],
                                                               feed_dict={self.all_real_data_int: _data,
                                                                          self.all_real_labels: _labels,
                                                                          self.new_noise: _data_HH,
                                                                          self._iteration: iteration,
                                                                          self.t_train: True})
                    _costs['cost'].append(_disc_cost)
                if cfg.MODE == 'wgan':
                    _ = self.session.run([clip_disc_weights])


            if iteration % cfg.SUMMARY_FREQUENCY == cfg.SUMMARY_FREQUENCY - 1:
                summary_writer.add_summary(_summary, iteration)

            if iteration % 100 == 99:
                _dev_cost_summary = self.session.run(dev_cost_summary, feed_dict={self.all_real_data_int: sample_images,
                                                                                  self.all_real_labels: sample_labels,
                                                                                  self.new_noise: sample_images_HH,
                                                                                  self.t_train: True})
                summary_writer.add_summary(_dev_cost_summary, iteration)
                generate_image0(self.cluster_0, iteration) # every 100iter save images from G.
                generate_image1(self.cluster_1, iteration) 
                generate_image2(self.cluster_2, iteration) 
                generate_image3(self.cluster_3, iteration) 


            if (iteration < 500) or (iteration % 1000 == 999):
                # ideally we have the averages here
                prints = 'iter %i' % iteration
                for name, values in _costs.items():
                    prints += "\t{}\t{}".format(name, np.mean(values))
                    _costs[name] = []
                print(prints)

            if iteration % 2000 == 1999:    # 100000 / 2000 // 5 = 10
                if iteration + 1 // 2000 % ((cfg.ITERS / 2000) // cfg.KEEP_CHECKPOINTS) == 0: # // 表示取整数部分，
                    # keep this checkpoint
                    self.saver.save(self.session, os.path.join(self.save_dir, 'model.ckpt'), global_step=iteration)
                else:
                    self.saver.save(self.session, os.path.join(self.save_dir, 'model.ckpt'))
                np.savetxt(os.path.join(self.save_dir, 'last_iteration'), [iteration])
                if cfg.bn_init:
                    cfg.bn_init = False
                    with open(os.path.join(self.run_dir, 'config.json'), 'w') as f:
                        json.dump(self.cfg.__dict__, f)
                print('Model saved.')

            if (cfg.INCEPTION_FREQUENCY != 0) and (iteration % cfg.INCEPTION_FREQUENCY == cfg.INCEPTION_FREQUENCY - 1):
                inception_score = get_inception_score(10000)
                print('INCEPTION SCORE:\t' + str(inception_score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=True) # 给train一个默认值True
    parser.add_argument('--load_config', action='store', type=str, default=None)
    config_dict = Config().__dict__
    for key, value in config_dict.iteritems():
        if key != 'train':
            parser.add_argument('--' + key, action='store', type=type(value), default=None) # 把config_dict中的配置放到parser中，后面传给args
    args = parser.parse_args()
    arg_dict = {}
    for arg in vars(args): # 返回对象object的属性和属性值的字典对象，如果没有参数，就打印当前调用位置的属性和属性值 类似 locals()。
        if arg != 'load_config':
            val = getattr(args, arg) # 返回args的arg属性的值
            if val is not None:
                arg_dict[arg] = val # 把Config类中的配置值传到arg_dict中

    with tf.Session() as session:
        if args.load_config is not None:
            wgan = WGAN(session, load_config=args.load_config)
        else:
            wgan = WGAN(session, config_dict=arg_dict, train=args.train)
        if args.train:
            wgan.train()

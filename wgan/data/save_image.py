import h5py
import numpy as np
import scipy.misc

def maybe_mkdirs(path): 
    # path can be a path list, so you can make a list of path.
    if type(path) is list:
        for path in path:
            maybe_mkdirs(path)
    else:
        if not os.path.exists(path):
            os.makedirs(path)

def merge(images,size):
    h,w,c = images.shape[1],images.shape[2],images.shape[3]
    img = np.zeros((h*size[0],w*size[1],c))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j= idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w,:] = image
    return img

def save_img(images,size,path):

    # if not os.path.exist(path):
    #     os.makedirs(path)
    # path = os.path.join(path,)
    return scipy.misc.imsave(path,merge(images,size))


if __name__ == '__main__':
    
    with h5py.File('LLD-icon-sharp.hdf5','r') as hdf_file:
        images = np.array(hdf_file['data']).transpose(0,2,3,1)
        label = np.array(hdf_file['labels/resnet1/rc_128'])
        class_indices = [np.where(label==i)[0] for i in range(128)]
        # sample = []
        for i in range(len(class_indices)):
            index_list = np.random.choice(class_indices[i],100)
            # sample.append(images[index_list]) # (128,100,32,32,3)
            sample_image = images[index_list]
            save_img(sample_image,size=(10,10),path='sample_{}.png'.format(i))
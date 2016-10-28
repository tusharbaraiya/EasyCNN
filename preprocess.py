import glob
from random import shuffle
from scipy import ndimage
import numpy as np
from scipy import misc

# function to generate list of subdirectories
def get_sub_dirs(path):
    return glob.glob(path + "/*/")

# function to get all images from all sun directories
def get_all_images_lables_list(sub_dirs):
    all_images_list=[]
    all_lables=[]
    label=0
    for sub in sub_dirs:
        sub_list = glob.glob(sub+"*")
        lablel_list = [label]*len(sub_list)
        all_lables.append(lablel_list)
        all_images_list.append(sub_list)
        label+=1
    return all_images_list,all_lables


# shuffle images and lablel list without losing correspondence
def random_shufle(image_list,lables_list):
    shuf_image_list=[]
    shuf_label_list=[]
    indexes = range(len(lables_list))
    shuffle(indexes)
    for i in indexes:
        shuf_image_list.append(image_list[i])
        shuf_label_list.append(lables_list[i])
    return shuf_image_list,shuf_label_list


# Function to split train and test data (split is 80:20)
def split_data(image_list,label_list):
    len = len(image_list)
    split = len*80/100
    train_images_list = image_list[0:split]
    train_label_list = label_list[0:split]
    test_images_list = image_list[split:len]
    test_label_list = label_list[split:len]
    return train_images_list,train_label_list,test_images_list,test_label_list


# Function to read and normlize images
def read_image_data(img_path,re_h,re_w):
    tensor = np.ndarray(shape=(re_h, re_w), dtype=np.float32)
    im = ndimage.imread(img_path, flatten=False).astype(float)
    im = misc.imresize(im, (re_h, re_w))
    image_data = (im - 255.0 / 2) / 255.0
    tensor[:, :] = image_data
    return tensor


# Function to generate image tensors
def generate_image_tensors(image_list,re_h,re_w):
    len = len(image_list)
    image_tensors = np.ndarray(shape=(len,re_h, re_w), dtype=np.float32)
    for i in range(len):
        image_tensors[i,:,:] = read_image_data(image_list[i],re_h,re_w)
    return image_tensors


# Function to gen one hot lable tensors
def gen_one_hot_lable_tensors(num_classes,label_list):
    label_tensors = np.ndarray(shape=(len(label_list),num_classes), dtype=np.float32)
    l = [0.0] * num_classes
    for i in range(len(label_list)):
        one_hot = np.array(l)
        one_hot[label_list[i]]=1.0
        label_tensors[i,:] = one_hot
    return label_tensors

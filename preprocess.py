import glob
from random import shuffle

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

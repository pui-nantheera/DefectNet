from __future__ import print_function

import os
import numpy as np
from PIL import Image
from skimage.io import imsave, imread
import warnings
from skimage.transform import resize
import tensorflow as tf

# Define flags
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("data_dir", "/mnt/storage/scratch/eexna/Palantir/Test_data_set/Test/", "path to dataset")
tf.flags.DEFINE_integer("patch_size", "200", "patch size for training")

# temp_dir = "/mnt/storage/scratch/eexna/Palantir/temp/"
temp_dir = "temp/"
try:
    # Create target Directory
    os.mkdir(temp_dir)
    print("Directory " , temp_dir ,  " Created ") 
except FileExistsError:
    print("Directory " , temp_dir ,  " already exists")

IMAGEROWS = 2000 #3632
IMAGECOLS = 3000 #5456

rows_patches = FLAGS.patch_size #800
cols_patches = FLAGS.patch_size #800

N_rows_pathches_for_prediction = np.int8(np.ceil(IMAGEROWS/rows_patches))
N_cols_pathches_for_prediction = np.int8(np.ceil(IMAGECOLS/cols_patches)) 

def image_to_mask(img):
    #imgs is a array of imgs with which catergory each pixrl is.
    
    #imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols,3), dtype=np.uint8)
    image_rows = img.shape[0]
    image_cols = img.shape[1]
    img_p = np.ndarray((image_rows, image_cols), dtype=np.uint8)+1
    colours = np.ndarray((3,3), dtype=np.uint8)
    
    #define colours
    colours[0] = [0,255,0]
    colours[1] = [255,0,0]
    colours[2] = [0,0,0]
    
    red,green = img[:,:,1],img[:,:,3]
    
    idx = red == 0
    img_p[idx] =  2
    
    idx = green == 0
    img_p[idx] =  3
    


    return img_p

def create_val_data():
    train_data_path = FLAGS.data_dir
    print(train_data_path)
    # images = os.listdir(train_data_path)

##    total = int(len(images)/3)*N_rows_pathches_for_prediction*N_cols_pathches_for_prediction
    #total = int(len(images)/2)*N_rows_pathches_for_prediction*N_cols_pathches_for_prediction
    images = [f for f in sorted(os.listdir(train_data_path)) if (f.endswith('.jpeg') or f.endswith('.JPG') or f.endswith('.jpg'))]
    total = len(images)*N_rows_pathches_for_prediction*N_cols_pathches_for_prediction
    #total = int(1000 / 3+1)

    imgs = np.ndarray((total, rows_patches, cols_patches,3), dtype=np.uint8)
    imgs_id = []
    
    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in images:
        if 'npy' in image_name or 'png' in image_name:
            continue
        img_id = image_name
        img = np.asarray(Image.open(os.path.join(train_data_path, image_name)))
        if (IMAGEROWS != img.shape[0]) | (IMAGECOLS != img.shape[1]):
            img = Image.open(os.path.join(train_data_path, image_name))
            img = img.resize((IMAGECOLS, IMAGEROWS), Image.ANTIALIAS)
            img = np.asarray(img)
            
        image_rows = img.shape[0]
        image_cols = img.shape[1]
        # print(image_name)
        #split image into 800 by 800
        image_number=i*N_rows_pathches_for_prediction*N_cols_pathches_for_prediction
        for x in range(0,N_rows_pathches_for_prediction):
            for y in range(0,N_cols_pathches_for_prediction):
                
                idx_row = int(x*image_rows/N_rows_pathches_for_prediction)
                idx_col = int(y*image_cols/N_cols_pathches_for_prediction)
                
                if idx_row+rows_patches>image_rows:
                    idx_row = image_rows-rows_patches
                if idx_col+cols_patches>image_cols:
                    idx_col = image_cols-cols_patches
                    
                img_sub = img[idx_row:idx_row+rows_patches,idx_col:idx_col+cols_patches]
        
                imgs[image_number] = img_sub
                imgs_id.append(img_id + '_x_' +str(x) + '_y_'+ str(y))
                
                image_number =image_number+1
        
        
        #if i % 1 == 0:
            #print('Done: {0}/{1} images'.format(i, len(images)/2))
        i += 1
    print('Loading done.')
    print(imgs.shape)
    np.save(temp_dir + 'imgs_val_full_size1.npy', imgs)
    np.save(temp_dir + 'imgs_id_val_full_size1.npy', imgs_id)
    print('Saving to .npy files done.')


def load_val_data():
    imgs_test = np.load(temp_dir + 'imgs_val_full_size1.npy')
    imgs_id = np.load(temp_dir + 'imgs_id_val_full_size1.npy')
    return imgs_test, imgs_id

if __name__ == '__main__':
    #create_val_data_npy()
    create_val_data()

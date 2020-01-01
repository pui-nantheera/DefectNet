import os
from skimage.transform import resize
# from skimage.io import imsave
import numpy as np
import PIL.Image as Image
import warnings
import scipy.misc as misc
import tensorflow as tf
import imageio

# Define flags
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("data_dir", "/mnt/storage/scratch/eexna/Palantir/Test_data_set/Test/", "path to dataset")
tf.flags.DEFINE_string("result_dir", "/mnt/storage/scratch/eexna/Palantir/Erosion_results/", "path to save predict result")
tf.flags.DEFINE_integer("patch_size", "200", "patch size for training")

# temp_dir = "/mnt/storage/scratch/eexna/Palantir/temp/"
# temp_dir = "../../results/temp/"
temp_dir = "temp/"

image_rows = 2000 #3632
image_cols = 3000 #5456

rows_patches = FLAGS.patch_size #800
cols_patches = FLAGS.patch_size #800

N_rows_pathches_for_prediction = np.int8(np.ceil(image_rows/rows_patches)) #10+3
N_cols_pathches_for_prediction = np.int8(np.ceil(image_cols/cols_patches)) #15+2

imgs_pred = np.load(FLAGS.result_dir + 'pred_800.npy')
imgs_test = np.load(temp_dir + 'imgs_val_full_size1.npy')
imgs_id_test = np.load(temp_dir + 'imgs_id_val_full_size1.npy')

def mask_to_image(imgs,orginal_img_rows,orginal_img_cols):
    #imgs is a array of imgs with which catergory each pixrl is.
    #imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols,3), dtype=np.uint8)
    imgs_p = np.ndarray((imgs.shape[0], orginal_img_rows, orginal_img_cols,3), dtype=np.uint8)
    imgs_noblade = np.ndarray((imgs.shape[0], orginal_img_rows, orginal_img_cols,3), dtype=np.uint8)
    colours = np.ndarray((14,3), dtype=np.uint8)
    #define colours
    colours[0] = [0, 0, 0] # Background
    colours[1] = [211, 199, 219] # Blades
    colours[2] = [255, 255, 0] # Contamination
    colours[3] = [255, 128, 64] # Dino tail
    colours[4] = [255, 128, 0]  # Drain Hole defects
    colours[5] = [30, 144, 255] #Erosion
    colours[6] = [64, 128, 128] #LPS
    colours[7] = [218, 165, 32]  #Pinholes
    colours[8] = [198, 230, 70] #Previous repair
    colours[9]= [36, 36, 255] #Scratches and Gouges
    colours[10]= [128, 0, 128] #Surface voids, Chipped paint, Peeling paint, Surface Delamination (1 area)
    colours[11]= [0, 191, 255] #Structural Cracks, Superficial Longitudinal crack,Superficial, Transverse crack, Trailing Edge Laminate Damage (1 area)
    colours[12]= [0, 64, 64]   #Vortex Generator
    colours[13] = [233, 233, 173]   # Unknown
    for i in range(imgs.shape[0]):
        imgs_p[i,:,:] =  colours[1]*0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img = np.round(resize(imgs[i], (orginal_img_rows, orginal_img_cols), preserve_range=True))
        for j in range(0,14):
            idx = img ==j
            imgs_p[i,idx,:] =  colours[j]
            if j != 1:
                imgs_noblade[i,idx,:] =  colours[j]
    return imgs_p, imgs_noblade


number_patches = imgs_test.shape[0]

print(np.shape(imgs_test))
print(np.shape(imgs_pred))
print(np.shape(imgs_id_test))

masks = np.zeros((int(number_patches/N_rows_pathches_for_prediction/N_cols_pathches_for_prediction),image_rows,image_cols)).astype('uint8')
imgs_id = []
    
image_number=0
for i in range(0,int(number_patches/N_rows_pathches_for_prediction/N_cols_pathches_for_prediction)):
    mask = np.zeros((image_rows,image_cols)).astype('uint8')
        
    for x in range(0,N_rows_pathches_for_prediction):
        for y in range(0,N_cols_pathches_for_prediction):
                
            idx = int(i*N_rows_pathches_for_prediction*N_cols_pathches_for_prediction+x*N_cols_pathches_for_prediction+y)
                
            mask_sub = imgs_pred[idx].astype('uint8')
                
            idx_row = int(x*image_rows/N_rows_pathches_for_prediction)
            idx_col = int(y*image_cols/N_cols_pathches_for_prediction)
                
            if idx_row+rows_patches>image_rows:
                idx_row = image_rows-rows_patches
            if idx_col+cols_patches>image_cols:
                idx_col = image_cols-cols_patches
                    
            mask[idx_row:idx_row+rows_patches,idx_col:idx_col+cols_patches] = mask_sub
            
    masks[image_number] = mask
    
    imgs_id.append("_".join(imgs_id_test[idx].split('_')[0:-4]))
        
    image_number =image_number+1

    # print(str(i+1) + '/' +str(int(number_patches/N_rows_pathches_for_prediction/N_cols_pathches_for_prediction)))
        
imgs_mask_full_size, imgs_noblade_full_size = mask_to_image(masks,image_rows,image_cols)
    
pred_dir = FLAGS.result_dir + 'preds_full_size'
if not os.path.exists(pred_dir):
    os.mkdir(pred_dir)
    
#for mask, image_id in zip(masks, imgs_id):
#    np.savez_compressed(os.path.join(pred_dir, str(image_id.replace('.jpg','.npy.npz')) ), mask)    
    
for image, image_noblade, image_id in zip(imgs_mask_full_size, imgs_noblade_full_size, imgs_id):
    mask = (image[:, :, :]).astype(np.uint8)
    # read raw blade image
    # rawimage = imageio.imread(os.path.join(FLAGS.data_dir,image_id))
    # rawimage = resize(rawimage, [image_rows, image_cols])*255
    # rawimage = rawimage*(image_noblade==0) + image_noblade
    # image = rawimage.astype(np.uint8)
    
    # imageio.imwrite(os.path.join(pred_dir, image_id), image)
    imageio.imwrite(os.path.join(pred_dir, str(image_id.replace('.jpg','.JPG')) + '.png'), mask)


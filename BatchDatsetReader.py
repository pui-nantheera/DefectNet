import numpy as np
import scipy.misc as misc
from random import shuffle
#import imageio

class BatchDatset:
    files = []
    images = []
    annotations = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, records_list, image_options={},num_class=9):
        """
        Intialize a generic file reader with batching for list of files
        :param records_list: list of file records to read -
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        :param image_options: A dictionary of options for modifying the output image
        Available options:
        resize = True/ False
        resize_size = #size of output image - does bilinear resize
        color=True/False
        """
        print("Initializing Batch Dataset Reader...")
        print(image_options)
        self.files = records_list
        self.image_options = image_options
        self.num_class = num_class
        # self._read_images()

    def _read_images(self, subFiles):
        self.__channels = True
        self.images = np.array([self._transform(filename['image']) for filename in subFiles])
        self.__channels = False
        self.annotations = np.array(
            [np.expand_dims(self._transform_annotation(filename['annotation']), axis=3) for filename in subFiles])
        # print (self.images.shape)
        # print (self.annotations.shape)

    def _transform_annotation(self, filename):
        b = misc.imread(filename)
        #b = imageio.imread(filename)
        # mask = np.logical_and(b[:,:,0]==30,np.logical_and(b[:,:,1]==144,b[:,:,2]==255))
        image = b-1
        if self.num_class==2:
            image = (image>1)*1

        if self.__channels and len(image.shape) < 3:  # make sure images are of shape(h,w,3)
            image = np.array([image for i in range(3)])

        if self.image_options.get("resize", False) and self.image_options["resize"]:
            resize_size = int(self.image_options["resize_size"])
            resize_image = misc.imresize(image,
                                         [resize_size, resize_size], interp='nearest')
        else:
            resize_image = image


        return np.array(resize_image)

    def _transform(self, filename):
        image = misc.imread(filename)
        #image = imageio.imread(filename)
        if self.__channels and len(image.shape) < 3:  # make sure images are of shape(h,w,3)
            image = np.array([image for i in range(3)])
            # image = np.transpose(image, axes=[1,2,0])

        if self.image_options.get("resize", False) and self.image_options["resize"]:
            resize_size = int(self.image_options["resize_size"])
            resize_image = misc.imresize(image,
                                         [resize_size, resize_size], interp='nearest')
        else:
            resize_image = image

        return np.array(resize_image)

    def get_records(self):
        return self.images, self.annotations

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > len(self.files):
            # Finished epoch
            self.epochs_completed += 1
            #print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # # Shuffle the data
            # perm = np.arange(len(self.files))
            # np.random.shuffle(perm)
            # self.files = self.files[perm]
            shuffle(self.files)
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        self._read_images(self.files[start:end])
        return self.images, self.annotations

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, len(self.files), size=[batch_size]).tolist()
        self._read_images(self.files[indexes])
        return self.images, self.annotations

import numpy as np
import os
import random
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
import glob

import TensorflowUtils as utils

DATA_URL = '.'

def read_dataset(data_dir):
    print ('read_dataset: ........................')
    pickle_filename = "BladeImageParsing.pickle"
    pickle_filepath = os.path.join(data_dir, pickle_filename)
    if not os.path.exists(pickle_filepath):
        SceneParsing_folder = os.path.splitext(DATA_URL.split("\\")[-1])[0]
        result = create_image_lists(os.path.join(data_dir, SceneParsing_folder))
        # print ("Pickling ..." + pickle_filepath)
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    else:
        print ("Found pickle file!")

    with open(pickle_filepath, 'rb') as f:
        result = pickle.load(f)
        training_records = result['training']
        validation_records = result['validation']
        test_records = result['test']
        del result

    return training_records, validation_records, test_records


def create_image_lists(image_dir):
    print ('Get images: ........................')
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    directories = ['training', 'validation', 'test']
    image_list = {}

    for directory in directories:
        file_list = []
        image_list[directory] = []
        #file_glob = os.path.join(image_dir, directory, '*.' + 'jpg')
        file_glob = os.path.join(image_dir, '*.' + 'jpg')
        file_list.extend(glob.glob(file_glob))

        if directory != 'training':
            file_list = file_list[::10]
        else:
            del file_list[::10]

        if not file_list:
            print('No files found')
        else:
            for f in file_list:
                filename = os.path.splitext(f.split("\\")[-1])[0]
#                annotation_file = os.path.join(image_dir, "annotations", directory, filename + '.npz')
                # annotation_file = os.path.join(image_dir, directory, filename + '.png')
                annotation_file = os.path.join(image_dir, filename + '.png')
                if os.path.exists(annotation_file):
                    record = {'image': f, 'annotation': annotation_file, 'filename': filename}
                    image_list[directory].append(record)
                else:
                    print("Annotation file not found for %s - Skipping" % filename)

#        random.shuffle(image_list[directory])
        no_of_images = len(image_list[directory])
        print ('No. of %s files: %d' % (directory, no_of_images))

    return image_list

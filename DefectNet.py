from __future__ import print_function
import tensorflow as tf
import numpy as np
# from scipy.misc import imsave
# import imageio
import scipy.ndimage as ndi
from PIL import Image, ImageDraw
import os

import TensorflowUtils as utils
import ReadBladeImageData as scene_parsing
import datetime
import BatchDatsetReader as dataset
from six.moves import xrange

# From Peter's code
from data_full_size import load_val_data

# Define flags
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "10", "batch size for training")
tf.flags.DEFINE_integer("patch_size", "200", "patch size for training")
tf.flags.DEFINE_string("logs_dir", "/Users/eexna/Work/Palantir/results/DefectNet_model/", "path to logs directory")
tf.flags.DEFINE_string("patch_dir", "/Users/eexna/Work/Palantir/data_external/Train_dataset/", "path to dataset")
tf.flags.DEFINE_string("result_dir", "/mnt/storage/scratch/eexna/Palantir/Erosion_results/", "path to save predict result")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "~/code_DefectNet/model/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")
tf.flags.DEFINE_integer("weight_type", "1", "Type of applying weights to classes: 0=uniform, 1=defined ratio")
tf.flags.DEFINE_integer("max_itr", "200000", "Maximum iteration in training")

# Use pretrained VGG model
MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

MAX_ITERATION = int(FLAGS.max_itr +1)  #int(2e5 + 1)
NUM_OF_CLASSESS = 14
IMAGE_SIZE = FLAGS.patch_size
patch_size = FLAGS.patch_size


def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

    return net

def vgg_dilate_net(weights, image):
    layers = (
        'convDilate1_1', 'reluDilate1_1', 'convDilate1_2', 'reluDilate1_2', 'poolDilate1',

        'convDilate2_1', 'reluDilate2_1', 'convDilate2_2', 'reluDilate2_2', 'poolDilate2',

        'convDilate3_1', 'reluDilate3_1', 'convDilate3_2', 'reluDilate3_2', 'convDilate3_3',
        'reluDilate3_3', 'convDilate3_4', 'reluDilate3_4', 

        'convDilate4_1', 'reluDilate4_1', 'convDilate4_2', 'reluDilate4_2', 'convDilate4_3',
        'reluDilate4_3', 'convDilate4_4', 'reluDilate4_4',

        'convDilate5_1', 'reluDilate5_1', 'convDilate5_2', 'reluDilate5_2', 'convDilate5_3',
        'reluDilate5_3', 'convDilate5_4', 'reluDilate5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            rate = int(name[10:11])
            if rate <= 2:
                kernels, bias = weights[i][0][0][0][0]
            elif rate > 2:
                kernels, bias = weights[7][0][0][0][0]
                if rate > 4:
                    rate = 3
                    if int(name[12:13]) == 4:
                        rate = 2
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.weight_variable(np.shape(np.transpose(kernels, (1, 0, 2, 3))), name=name + "_wDilate")
            bias = utils.bias_variable(np.shape(bias.reshape(-1)), name=name + "_bDilate")
            current = utils.conv2d_dilate(current, 2**(rate-1), kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

    return net

def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])

    processed_image = utils.process_image(image, mean_pixel)

    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image)
        imageDilate_net = vgg_dilate_net(weights, processed_image)
        conv_final_layer = image_net["conv5_3"]
        convDilate_final_layer = imageDilate_net["convDilate5_3"]

        pool5 = utils.max_pool_2x2(conv_final_layer)

        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)

        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        deconv_shape3 = image_net["pool2"].get_shape()
        W_t3 = utils.weight_variable([4, 4, deconv_shape3[3].value, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([deconv_shape3[3].value], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=tf.shape(image_net["pool2"]))
        fuse_3a = tf.add(conv_t3, image_net["pool2"], name="fuse_3a")
        fuse_3 = tf.add(fuse_3a, imageDilate_net["reluDilate5_4"], name="fuse_3")

        deconv_shape4 = image_net["pool1"].get_shape()
        W_t4 = utils.weight_variable([4, 4, deconv_shape4[3].value, deconv_shape3[3].value], name="W_t4")
        b_t4 = utils.bias_variable([deconv_shape4[3].value], name="b_t4")
        conv_t4 = utils.conv2d_transpose_strided(fuse_3, W_t4, b_t4, output_shape=tf.shape(image_net["pool1"]))
        fuse_4 = tf.add(conv_t4, image_net["pool1"], name="fuse_4")

        shape = tf.shape(image)
        deconv_shape5 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t5 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape4[3].value], name="W_t5")
        b_t5 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t5")
        conv_t5 = utils.conv2d_transpose_strided(fuse_4, W_t5, b_t5, output_shape=deconv_shape5, stride=2)

        annotation_pred = tf.argmax(conv_t5, dimension=3, name="prediction")

    return tf.expand_dims(annotation_pred, dim=3), conv_t5

def labels_to_one_hot(ground_truth, num_classes=1):
    """
    Converts ground truth labels to one-hot, sparse tensors.
    Used extensively in segmentation losses.
    :param ground_truth: ground truth categorical labels (rank `N`)
    :param num_classes: A scalar defining the depth of the one hot dimension
        (see `depth` of `tf.one_hot`)
    :return: one-hot sparse tf tensor
        (rank `N+1`; new axis appended at the end)
    """
    # read input/output shapes
    if isinstance(num_classes, tf.Tensor):
        num_classes_tf = tf.to_int32(num_classes)
    else:
        num_classes_tf = tf.constant(num_classes, tf.int32)
    input_shape = tf.shape(ground_truth)
    output_shape = tf.concat(
        [input_shape, tf.reshape(num_classes_tf, (1,))], 0)
    if num_classes == 1:
        # need a sparse representation?
        return tf.reshape(ground_truth, output_shape)
    # squeeze the spatial shape
    ground_truth = tf.reshape(ground_truth, (-1,))
    # shape of squeezed output
    dense_shape = tf.stack([tf.shape(ground_truth)[0], num_classes_tf], 0)
    # create a rank-2 sparse tensor
    ground_truth = tf.to_int64(ground_truth)
    ids = tf.range(tf.to_int64(dense_shape[0]), dtype=tf.int64)
    ids = tf.stack([ids, ground_truth], axis=1)
    one_hot = tf.SparseTensor(
        indices=ids,
        values=tf.ones_like(ground_truth, dtype=tf.float32),
        dense_shape=tf.to_int64(dense_shape))
    # resume the spatial dims
    one_hot = tf.sparse_reshape(one_hot, output_shape)
    return one_hot


def generalised_dice_loss(prediction,
                          ground_truth,
                          weight_map=None,
                          type_weight='Square'):
    """
    Function to calculate the Generalised Dice Loss defined in
        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017
    :param prediction: the logits
    :param ground_truth: the segmentation ground truth
    :param weight_map:
    :param type_weight: type of weighting allowed between labels (choice
        between Square (square of inverse of volume),
        Simple (inverse of volume) and Uniform (no weighting))
    :return: the loss
    """
    prediction = tf.cast(prediction, tf.float32)
    if len(ground_truth.shape) == len(prediction.shape):
        ground_truth = ground_truth[..., -1]
    one_hot = labels_to_one_hot(ground_truth, tf.shape(prediction)[-1])
    if weight_map is not None:
        n_classes = prediction.shape[1].value
        weight_map_nclasses = tf.reshape(
            tf.tile(weight_map, [n_classes]), prediction.get_shape())
        ref_vol = tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot, reduction_axes=[0])
        intersect = tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot * prediction, reduction_axes=[0])
        seg_vol = tf.reduce_sum(
            tf.multiply(weight_map_nclasses, prediction), 0)
    else:
        ref_vol = tf.sparse_reduce_sum(one_hot, [0,1,2])
        intersect = tf.sparse_reduce_sum(one_hot * prediction, [0,1,2])
        seg_vol = tf.reduce_sum(prediction, [0,1,2])
    if type_weight == 'Square':
        weights = tf.reciprocal(tf.square(ref_vol))
    elif type_weight == 'Simple':
        weights = tf.reciprocal(ref_vol)
    elif type_weight == 'Uniform':
        weights = tf.ones_like(ref_vol)
    elif type_weight =='Fixed':
        weights = tf.constant([0.0006, 0.0006, 0.1656, 0.1058, 0.0532, 0.0709, 0.1131, 0.3155, 0.1748])  #W3 = 1/sqrt(freq)
    else:
        raise ValueError("The variable type_weight \"{}\""
                         "is not defined.".format(type_weight))
    new_weights = tf.where(tf.is_inf(weights), tf.zeros_like(weights), weights)
    weights = tf.where(tf.is_inf(weights), tf.ones_like(weights) * tf.reduce_max(new_weights), weights)

    generalised_dice_numerator = 2 * tf.reduce_sum(tf.multiply(weights, intersect))
    generalised_dice_denominator = tf.reduce_sum(tf.multiply(weights, tf.maximum(seg_vol + ref_vol, 1)))
    # generalised_dice_denominator = tf.reduce_sum(tf.multiply(weights, seg_vol + ref_vol)) + 1e-6
    generalised_dice_score = generalised_dice_numerator / generalised_dice_denominator
    generalised_dice_score = tf.where(tf.is_nan(generalised_dice_score), 1.0, generalised_dice_score)

    return 1 - generalised_dice_score

def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    return optimizer.apply_gradients(grads)

def targets_to_rgb(target_array):
    im = np.zeros(target_array.shape + (3,), dtype='int8')
    im[:, :, 1][target_array == 1] = 255
    im[:, :, 0][target_array == 2] = 255
    im[:, :, :][target_array == 3] = [255, 255, 255]
    return Image.fromarray(im, 'RGB')

def main(argv=None):

    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")

    pred_annotation, logits = inference(image, keep_probability)
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)
    labels = tf.squeeze(annotation, squeeze_dims=[3])
    if FLAGS.weight_type==1:
        class_weight = tf.constant([0.0013,    0.0012,    0.0305,    0.0705,    0.8689,    0.2045,    0.0937,    0.8034,    0.0046,    0.1884,    0.1517,    0.5828,    0.0695, 0.0019]) #1/freq^(1/3)
        weights = tf.gather(class_weight, labels)
        loss = tf.reduce_mean((tf.losses.sparse_softmax_cross_entropy(logits=logits,
                                                                          labels=labels,
                                                                          weights=weights)))
    elif FLAGS.weight_type==2:
        loss = generalised_dice_loss(prediction=logits,ground_truth=labels,type_weight='Fixed')
        # loss = tversky_loss(labels, logits)
    elif FLAGS.weight_type==3:
        loss = weight_generalised_dice_loss(prediction=logits,ground_truth=labels)
    elif FLAGS.weight_type==4:
        class_weight = tf.constant([0.0013,    0.0012,    0.0305,    0.0705,    0.8689,    0.2045,    0.0937,    0.8034,    0.0046,    0.1884,    0.1517,    0.5828,    0.0695, 0.0019]) #1/freq^(1/3)
        weights = tf.gather(class_weight, labels)
        loss1 = tf.reduce_mean((tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels, weights=weights)))
        loss2 = generalised_dice_loss(prediction=logits,ground_truth=labels)
        loss = tf.minimum(loss1,loss2)
    else:
        loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=labels,
                                                                          name="entropy")))
    tf.summary.scalar("entropy", loss)

    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss, trainable_var)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()


    # print(train_records)
    print("Setting up dataset reader")
    image_options = {'resize': False, 'resize_size': IMAGE_SIZE}
    if FLAGS.mode == 'train':
        train_records, valid_records, test_records = scene_parsing.read_dataset(FLAGS.patch_dir)
        print(len(train_records))
        print(len(valid_records))
        train_dataset_reader = dataset.BatchDatset(train_records, image_options)
        validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)
        test_dataset_reader = dataset.BatchDatset(test_records, image_options)

    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)
    print("Setting up global_variables_initializer...")
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    model_name = [f for f in os.listdir(FLAGS.logs_dir) if ("ckpt" in f)]
    if len(model_name)>0:
        ##  saver.restore(sess, ckpt.model_checkpoint_path)
        # saver.restore(sess, FLAGS.logs_dir + model_name[2][:model_name[2].find("meta")-1])
        # print("Model restored..." + model_name[2][:model_name[2].find("meta")-1])
        model_name = [f for f in model_name if ("meta" in f)]
        saver.restore(sess, FLAGS.logs_dir + model_name[0][:model_name[0].find("meta")-1])
        print("Model restored..." + model_name[0][:model_name[0].find("meta")-1])

    print('Processing ' + FLAGS.mode)
    if FLAGS.mode == "train":
        for itr in xrange(MAX_ITERATION):
            train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}

            sess.run(train_op, feed_dict=feed_dict)

            if itr % 50 == 0:
                train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                summary_writer.add_summary(summary_str, itr)

            if itr % 500 == 0:
                valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
                valid_loss = sess.run(loss, feed_dict={image: valid_images, annotation: valid_annotations,
                                                       keep_probability: 1.0})
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)

    elif FLAGS.mode == "evaluate":
#        test_images, test_annotations = test_dataset_reader.get_random_batch(FLAGS.batch_size)

        test_images, test_annotations = load_val_data()
        number_patches = test_images.shape[0]
        test_images1 = np.ndarray((1, patch_size, patch_size, 3), dtype=np.uint8)
        pred_800 = np.ndarray((number_patches, patch_size, patch_size), dtype=np.uint8)

        print ("Shapes of images:")
        print (np.shape(test_images))
        print (np.shape(test_annotations))

        for i in range(0, number_patches):
            test_images1[0,:,:,:] = test_images[i,:,:,:]
            pred = sess.run(pred_annotation, feed_dict={image: test_images1, keep_probability: 1.0})
            pred = np.squeeze(pred, axis=3)
            pred_800[i] = ndi.zoom(pred[0], patch_size/IMAGE_SIZE, order=0)
            print(datetime.datetime.now())
            # print("sum 0: %g"  %(pred==0).sum() + " sum 1: %g"  %(pred==1).sum())

        np.save(FLAGS.result_dir + "pred_800" + ".npy", pred_800)

if __name__ == "__main__":
    tf.app.run()
    print('Done' + FLAGS.mode)

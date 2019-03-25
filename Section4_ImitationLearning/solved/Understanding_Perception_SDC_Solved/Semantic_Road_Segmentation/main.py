import time
import os.path
import helper
import scipy.misc
import tensorflow as tf
from datetime import timedelta
from distutils.version import LooseVersion


L2_REG = 1e-6
STDEV = 1e-3
KEEP_PROB = 0.5
LEARNING_RATE = 1e-4
EPOCHS = 30
BATCH_SIZE = 16
IMAGE_SHAPE = (256, 512)
NUM_CLASSES = 3

DATA_DIR = './data'
RUNS_DIR = './output'
MODEL_DIR = './model'


def load_vgg(sess, vgg_path):

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    graph = tf.get_default_graph()
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    return input, keep_prob, layer3, layer4, layer7

print("Load VGG Model:")


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):

    layer7_conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, 1,
                                       padding='same', kernel_initializer=tf.random_normal_initializer(stddev=STDEV),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG))
    output = tf.layers.conv2d_transpose(layer7_conv_1x1, num_classes, 4, 2,
                                        padding='same', kernel_initializer=tf.random_normal_initializer(stddev=STDEV),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG))
    layer4_conv_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, 1,
                                       padding='same', kernel_initializer=tf.random_normal_initializer(stddev=STDEV),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG))
    output = tf.add(output, layer4_conv_1x1)
    output = tf.layers.conv2d_transpose(output, num_classes, 4, 2,
                                        padding='same', kernel_initializer=tf.random_normal_initializer(stddev=STDEV),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG))
    layer3_conv_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, 1,
                                       padding='same', kernel_initializer=tf.random_normal_initializer(stddev=STDEV),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG))
    output = tf.add(output, layer3_conv_1x1)
    output = tf.layers.conv2d_transpose(output, num_classes, 16, 8,
                                        padding='same', kernel_initializer=tf.random_normal_initializer(stddev=STDEV),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG))
    return output

print("Layers Test:")

weights = [0.3, 0.6, 0.3]

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    softmax = tf.nn.softmax(logits)
    cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax), weights), reduction_indices=[1])
    cross_entropy_loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss

print("Optimize Test:")


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, saver, data_dir):

    for epoch in range(epochs):
        s_time = time.time()
        for image, targets in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                feed_dict = {input_image: image, correct_label: targets, keep_prob: KEEP_PROB ,
                             learning_rate: LEARNING_RATE }) #/ (epoch/100 + 1)
            print(loss)
        # Print data on the learning process
        print("Epoch: {}".format(epoch + 1), "/ {}".format(epochs), " Loss: {:.3f}".format(loss), " Time: ",
              str(timedelta(seconds=(time.time() - s_time))))
        if (epoch + 1) % 10 == 0: # Save every 10 epochs
            save_path = saver.save(sess, os.path.join(data_dir, 'cont_epoch_' + str(epoch) + '.ckpt'))



def run():
    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(DATA_DIR)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/
    print("Start training...")
    config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5))
    with tf.Session(config=config) as sess:
        # Path to vgg model
        vgg_path = os.path.join(DATA_DIR, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(DATA_DIR, 'leftImg8bit'), IMAGE_SHAPE)
        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        # Add some augmentations, see helper.py
        input, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        output = layers(layer3, layer4, layer7, NUM_CLASSES)
        correct_label = tf.placeholder(dtype = tf.float32, shape = (None, None, None, NUM_CLASSES))
        learning_rate = tf.placeholder(dtype = tf.float32)
        logits, train_op, cross_entropy_loss = optimize(output, correct_label, learning_rate, NUM_CLASSES)
        tf.set_random_seed(123)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver() #Simple model saver
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, input, correct_label,
                 keep_prob, learning_rate,  saver, MODEL_DIR)
        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(RUNS_DIR, DATA_DIR, sess, IMAGE_SHAPE, logits, keep_prob, input, NUM_CLASSES)



def save_samples():
    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(DATA_DIR, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(DATA_DIR, 'data_road/training'), IMAGE_SHAPE)
        input, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        output = layers(layer3, layer4, layer7, NUM_CLASSES)
        correct_label = tf.placeholder(dtype = tf.float32, shape = (None, None, None, NUM_CLASSES))
        learning_rate = tf.placeholder(dtype = tf.float32)
        logits, train_op, cross_entropy_loss = optimize(output, correct_label, learning_rate, NUM_CLASSES)
        sess.run(tf.global_variables_initializer())
        new_saver = tf.train.import_meta_graph('./models_3col/epoch_199.ckpt.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('./models_3col/'))
        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(RUNS_DIR, DATA_DIR, sess, IMAGE_SHAPE, logits, keep_prob, input, NUM_CLASSES)


def cont():
    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(DATA_DIR, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(DATA_DIR, 'data_road/training'), IMAGE_SHAPE)
        input, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        output = layers(layer3, layer4, layer7, NUM_CLASSES)
        correct_label = tf.placeholder(dtype = tf.float32, shape = (None, None, None, NUM_CLASSES))
        learning_rate = tf.placeholder(dtype = tf.float32)
        logits, train_op, cross_entropy_loss = optimize(output, correct_label, learning_rate, NUM_CLASSES)
        sess.run(tf.global_variables_initializer())
        new_saver = tf.train.import_meta_graph('./models_3col/epoch_199.ckpt.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('./models_3col/'))
        saver = tf.train.Saver() #Simple model saver
        train_nn(sess, 10, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, input, correct_label,
                 keep_prob, learning_rate,  saver, MODEL_DIR)
        helper.save_inference_samples(RUNS_DIR, DATA_DIR, sess, IMAGE_SHAPE, logits, keep_prob, input, NUM_CLASSES)

if __name__ == "__main__":
    run()
    save_samples()
    cont()

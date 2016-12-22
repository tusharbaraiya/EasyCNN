import argparse
import preprocess as pre
import tensorflow as tf
import numpy as np


learning_rate = 0.001
training_epoches = 20
batch_size = 100

image_h = 28
image_w = 28
input_channels = 1
n_classes = 10
depth = 2


def train():
    x = tf.placeholder(tf.float32, [None, image_w, image_h])
    y = tf.placeholder(tf.float32, [None, n_classes])
    pred = model(x,depth)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        train_dataset = np.load("train_images.npy")
        test_dataset = np.load("test_images.npy")
        train_labels = np.load("train_labels.npy")
        test_labels = np.load("test_labels.npy")
        train_num = train_dataset.shape[0]
        num_batches = train_num/batch_size
        for i in range(training_epoches):
            print "epoch num :" + str(i+1)
            for j in range(num_batches-1):
                print "batch :" + str(j + 1)
                if train_num < (j + 1)*batch_size:
                    break
                batch_x = train_dataset[j * batch_size: (j + 1)*batch_size, :, :]
                batch_y = train_labels[j * batch_size: (j + 1)*batch_size, :]
                sess.run(optimizer, feed_dict={x: batch_x,
                                               y: batch_y})


def model(input,n):
    in_channel = 1
    out_feature = 32
    batch_n = batch_size
    n_class = n_classes
    kernel = tf.Variable(tf.random_normal([5, 5, in_channel, out_feature]))
    input = input[:,:,:,None]
    conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.zeros([out_feature], dtype=tf.float32))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME')
    norm = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    for i in range(n-1):
        kernel = tf.Variable(tf.random_normal([5, 5, out_feature,out_feature+32]))
        conv = tf.nn.conv2d(norm, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.zeros([out_feature+32],dtype=tf.float32))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv_n = tf.nn.relu(pre_activation)
        pool_n = tf.nn.max_pool(conv_n, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding='SAME')
        norm = tf.nn.lrn(pool_n, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        out_feature += 32

    reduced_n = image_w / (2**n)
    reshape_layer_num = (reduced_n**2)*out_feature
    reshape_layer = tf.reshape(norm, [batch_n,-1])
    fc1_weights = tf.Variable(tf.random_normal([reshape_layer_num,400]))
    fc1_biases = tf.Variable(tf.zeros([400]))
    fc1 = tf.nn.relu(tf.matmul(reshape_layer,fc1_weights) + fc1_biases)


    fc2_weights = tf.Variable(tf.random_normal([400, 200]))
    fc2_biases = tf.Variable(tf.zeros([200]))
    fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)

    out_weights = tf.Variable(tf.random_normal([200,n_class]))
    out_biases = tf.Variable(tf.zeros([n_class]))
    softmax = tf.add(tf.matmul(fc2, out_weights), out_biases)

    return softmax


def main():
    global image_h,image_w,learning_rate,training_epoches,batch_size,n_classes,depth
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', '--data-path', type=str, required=True)
    parser.add_argument('-image_re_h', '--image-reh', type=int, required=True)
    parser.add_argument('-image_re_w', '--image-rew', type=int, required=True)
    parser.add_argument('-batch_size', '--batch-size', type=int, required=True)
    parser.add_argument('-depth', '--depth', type=int, required=True)
    parser.add_argument('-learning_rate', '--learning-rate', type=float)
    parser.add_argument('-epoches', '--epoches', type=int)
    parser.add_argument('-mode', '--mode', type=str)
    args = parser.parse_args()
    #pre.gen_and_serialize(args.data_path, args.image_reh,args.image_rew)
    batch_size = args.batch_size
    image_h = args.image_reh
    image_w = args.image_rew
    depth = args.depth
    learning_rate = args.learning_rate
    training_epoches = args.epoches
    n_classes = pre.get_class_count(args.data_path)
    train()

main()

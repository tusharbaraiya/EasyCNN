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


def init_params(image_height=28,image_width=28,channels=1,dropout_val=0.75,
                learning_rate_val = 0.001,training_epoches_count = 20,batch_size_count = 100):
    global image_h,image_w,input_channels,dropout,learning_rate,training_epoches,batch_size
    image_h = image_height
    image_w = image_width
    input_channels=channels
    dropout = dropout_val
    learning_rate = learning_rate_val
    training_epoches = training_epoches_count
    batch_size = batch_size_count
    print "Initialized Parameters!"


def load_datasets():
    global train_dataset,test_dataset,train_labels,test_labels,train_size
    train_dataset = np.load("train_images.npy")
    test_dataset = np.load("test_images.npy")
    train_labels = np.load("train_labels.npy")
    test_labels = np.load("test_labels.npy")
    train_size = train_dataset.shape[0]


weights = {
    'wc1':tf.Variable(tf.random_normal([5,5,1,32])),
    'wc2':tf.Variable(tf.random_normal([5,5,32,64])),
    'wd1':tf.Variable(tf.random_normal([7*7*64,1024])),
    'out':tf.Variable(tf.random_normal([1024,n_classes]))
}


biases = {
    'bc1':tf.Variable(tf.random_normal([32])),
    'bc2':tf.Variable(tf.random_normal([64])),
    'bd1':tf.Variable(tf.random_normal([1024])),
    'out':tf.Variable(tf.random_normal([n_classes]))
}


#placeholders
x = tf.placeholder(tf.float32,[None,image_w,image_h])
y = tf.placeholder(tf.float32,[None,n_classes])
keep_prob = tf.placeholder(tf.float32)


def conv2d(x,W,b,strides=1):
    x = tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding='SAME')
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)


def maxpool2d(x,k=2):
    return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')

def create_model(x,weights,biases,dropout):
    x = tf.reshape(x,shape=[-1,28,28,1])
    conv1 = conv2d(x,weights['wc1'],biases['bc1'])
    conv1 = maxpool2d(conv1,k=2)

    conv2 = conv2d(conv1,weights['wc2'],biases['bc2'])
    conv2 = maxpool2d(conv2,k=2)

    fc1 = tf.reshape(conv2, [-1,weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1,weights['wd1']),biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1,dropout)
    out = tf.add(tf.matmul(fc1,weights['out']),biases['out'])
    return out

pred = create_model(x,weights,biases,keep_prob)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def start_training():
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        step = 0
        epoch_count = 0
        roll_over = train_size/batch_size
        print "Training started"
        while epoch_count < training_epoches:
            batch_count = (step+1)*batch_size
            batch_x = train_dataset[batch_count-batch_size:batch_count]
            batch_y = train_labels[batch_count-batch_size:batch_count]
            sess.run(optimizer,feed_dict={ x : batch_x,
                                           y : batch_y,
                                           keep_prob: 1.
            })
            step += 1
            step %= roll_over
            if step==0:
                epoch_count += 1
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                                  y: batch_y,
                                                                  keep_prob: 1.})
                print "Finished Epoch : "+str(epoch_count)+\
                      ", Minibatch Loss = " + str(loss)+\
                      ", Minibatch accuracy = " + str(acc)

        print ", test accuracy :", \
            sess.run(accuracy, feed_dict={x: test_dataset,
                                          y: test_labels,
                                          keep_prob: 1.
                                          })


def create_model(input,n):
    in_channel = 1
    out_feature = 32
    batch_n = batch_size
    n_class = 10
    kernel = tf.Variable(tf.random_normal([5, 5, in_channel, out_feature]))
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
    fc1_biases = tf.Variable(tf.constant_initializer([400],value=0.1))
    fc1 = tf.nn.relu(tf.matmul(reshape_layer,fc1_weights) + fc1_biases)


    fc2_weights = tf.Variable(tf.random_normal([400, 200]))
    fc2_biases = tf.Variable(tf.constant_initializer([200],value=0.1))
    fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)

    out_weights = tf.Variable(tf.random_normal([200,n_class]))
    out_biases = tf.Variable(tf.constant_initializer([n_class],value=0.1))
    softmax = tf.add(tf.matmul(fc2, out_weights), out_biases)

    return softmax


def main():
    global image_h,image_w,learning_rate,training_epoches,batch_size
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', '--data-path', type=str, required=True)
    parser.add_argument('-image_re_h', '--image-reh', type=int, required=True)
    parser.add_argument('-image_re_w', '--image-rew', type=int, required=True)
    parser.add_argument('-batch_size', '--batch-size', type=int, required=True)
    parser.add_argument('-learning_rate', '--learning-rate', type=float)
    parser.add_argument('-epoches', '--epoches', type=int)
    parser.add_argument('-mode', '--mode', type=str)
    args = parser.parse_args()
    pre.gen_and_serialize(args.data_path, args.image_reh,args.image_rew)
    batch_size = args.batch_size
    image_h = args.image_reh
    image_w = args.image_rew
    learning_rate = args.learning_rate
    training_epoches = args.epoches



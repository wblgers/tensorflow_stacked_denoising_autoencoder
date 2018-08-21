import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from library.Autoencoder import Autoencoder

# Read database
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyper parameters
n_samples = int(mnist.train.num_examples)
training_epochs = 20
batch_size = 128
display_step = 1

corruption_level = 0.3
sparse_reg = 0

#
n_inputs = 784
n_hidden = 400
n_hidden_fc = 784
n_outputs = 10
lr = 0.001

# define the autoencoder
ae = Autoencoder(n_layers=[n_inputs, n_hidden],
                          transfer_function = tf.nn.relu,
                          optimizer = tf.train.AdamOptimizer(learning_rate = lr),
                          ae_para = [corruption_level, sparse_reg])

# define the output layer using softmax
xavier_initializer = tf.contrib.layers.xavier_initializer()
x = tf.placeholder(tf.float32, [None, n_hidden])
W = tf.Variable(xavier_initializer((n_hidden, n_hidden_fc),dtype=tf.float32))
b = tf.Variable(tf.zeros([n_hidden_fc]))
h = tf.nn.relu(tf.matmul(x, W) + b)

W_out = tf.Variable(xavier_initializer((n_hidden_fc, n_outputs),dtype=tf.float32))
b_out = tf.Variable(tf.zeros([n_outputs]))

# y = tf.matmul(x, W) + b
y = tf.matmul(h, W_out) + b_out

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, n_outputs])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
train_step = tf.train.AdamOptimizer(learning_rate = lr).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

## define the output layer using softmax in the fine tuning step
x_ae = ae.transform()
h_ft = tf.nn.relu(tf.matmul(x_ae, W) + b)
# y_ft = tf.matmul(x_ae, W) + b
y_ft = tf.matmul(h_ft, W_out) + b_out

cross_entropy_ft = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_ft, labels=y_))+\
                   tf.reduce_mean(tf.contrib.layers.l2_regularizer(1e-4)(W))+\
                   tf.reduce_mean(tf.contrib.layers.l2_regularizer(1e-4)(W_out))
train_step_ft = tf.train.AdamOptimizer(learning_rate = lr).minimize(cross_entropy_ft)

correct_prediction_ft = tf.equal(tf.argmax(y_ft, 1), tf.argmax(y_, 1))
accuracy_ft = tf.reduce_mean(tf.cast(correct_prediction_ft, tf.float32))

## Initialization
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(n_samples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs, _ = mnist.train.next_batch(batch_size)
        # Fit training using batch data
        temp = ae.partial_fit()
        cost, opt = sess.run(temp,feed_dict={ae.x: batch_xs, ae.keep_prob : ae.in_keep_prob})

        # Compute average loss
        avg_cost += cost / n_samples * batch_size

    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%d,' % (epoch + 1),
              "Cost:", "{:.9f}".format(avg_cost))
ae_test_cost = sess.run(ae.calc_total_cost(),feed_dict={ae.x: mnist.test.images, ae.keep_prob : 1.0})
print("Total cost: " + str(ae_test_cost))

print("************************Finish the autoencoder training******************************")

# Training the softmax layer
for i in range(5000):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    x_ae = sess.run(ae.transform(),feed_dict={ae.x: batch_xs, ae.keep_prob : ae.in_keep_prob})
    if i%100 == 0:
        print(sess.run(accuracy, feed_dict={x: x_ae, y_: batch_ys,
                                               ae.keep_prob: 1.0}))
    sess.run(train_step, feed_dict={x: x_ae, y_: batch_ys})

print("************************Finish the softmax output layer training******************************")
print("Test accuracy before fine-tune")
print(sess.run(accuracy_ft, feed_dict={ae.x: mnist.test.images, y_: mnist.test.labels,
                                    ae.keep_prob: 1.0}))

# Training of fine tune
for i in range(5000):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    if i%100 == 0:
        print(sess.run(accuracy_ft, feed_dict={ae.x: batch_xs, y_: batch_ys,
                                               ae.keep_prob: 1.0}))
    sess.run(train_step_ft,feed_dict={ae.x: batch_xs, y_: batch_ys, ae.keep_prob : ae.in_keep_prob})
print("************************Finish the fine tuning******************************")
# Test trained model
print(sess.run(accuracy_ft, feed_dict={ae.x: mnist.test.images, y_: mnist.test.labels, ae.keep_prob : 1.0}))
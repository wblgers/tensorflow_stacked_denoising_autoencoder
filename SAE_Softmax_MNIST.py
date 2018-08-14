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
n_hidden2 = 100
n_outputs = 10
lr = 0.001

# define the autoencoder
ae = Autoencoder(n_layers=[n_inputs, n_hidden],
                          transfer_function = tf.nn.relu,
                          optimizer = tf.train.AdamOptimizer(learning_rate = lr),
                          ae_para = [corruption_level, sparse_reg])
ae_2nd = Autoencoder(n_layers=[n_hidden, n_hidden2],
                          transfer_function = tf.nn.relu,
                          optimizer = tf.train.AdamOptimizer(learning_rate = lr),
                          ae_para=[corruption_level, sparse_reg])
# define the output layer using softmax
x = tf.placeholder(tf.float32, [None, n_hidden2])
W = tf.Variable(tf.zeros([n_hidden2, n_outputs]))
b = tf.Variable(tf.zeros([n_outputs]))
y = tf.matmul(x, W) + b

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, n_outputs])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
train_step  = tf.train.AdamOptimizer(learning_rate = lr).minimize(cross_entropy)

## define the output layer using softmax in the fine tuning step
x_ft = tf.placeholder(tf.float32, [None, n_inputs])
h = x_ft

# Go through the two autoencoders
for layer in range(len(ae.n_layers) - 1):
    # h = tf.nn.dropout(h, ae.in_keep_prob)
    h = ae.transfer(
        tf.add(tf.matmul(h, ae.weights['encode'][layer]['w']),ae.weights['encode'][layer]['b']))
for layer in range(len(ae_2nd.n_layers) - 1):
    # h = tf.nn.dropout(h, ae_2nd.in_keep_prob)
    h = ae_2nd.transfer(
        tf.add(tf.matmul(h, ae_2nd.weights['encode'][layer]['w']),ae_2nd.weights['encode'][layer]['b']))

y_ft = tf.matmul(h, W) + b
cross_entropy_ft = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_ft, labels=y_))

train_step_ft  = tf.train.AdamOptimizer(learning_rate = lr).minimize(cross_entropy_ft)
correct_prediction = tf.equal(tf.argmax(y_ft, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

## Initialization
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs,_ = mnist.train.next_batch(batch_size)

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

print("************************First AE training finished******************************")


for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs, _ = mnist.train.next_batch(batch_size)

        # Fit training using batch data
        h_ae1_out = sess.run(ae.transform(),feed_dict={ae.x: batch_xs, ae.keep_prob : ae.in_keep_prob})
        temp = ae_2nd.partial_fit()
        cost, opt = sess.run(temp,feed_dict={ae_2nd.x: h_ae1_out, ae_2nd.keep_prob : ae_2nd.in_keep_prob})

        # Compute average loss
        avg_cost += cost / n_samples * batch_size

    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%d,' % (epoch + 1),
              "Cost:", "{:.9f}".format(avg_cost))

print("************************Second AE training finished******************************")


# Training the softmax layer
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    h_ae1_out = sess.run(ae.transform(), feed_dict={ae.x: batch_xs, ae.keep_prob : 1.0})
    h_ae2_out = sess.run(ae_2nd.transform(), feed_dict={ae_2nd.x: h_ae1_out, ae_2nd.keep_prob : 1.0})
    sess.run(train_step, feed_dict={x: h_ae2_out, y_: batch_ys})
print("*************************Finish the softmax output layer training*****************************")

print("Test accuracy before fine-tune")
print(sess.run(accuracy, feed_dict={x_ft: mnist.test.images, y_: mnist.test.labels,
                                    ae.keep_prob: 1.0,ae_2nd.keep_prob : 1.0}))

# Training of fine tune
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    sess.run(train_step_ft,feed_dict={x_ft: batch_xs, y_: batch_ys,
                                      ae.keep_prob: 1.0, ae_2nd.keep_prob: 1.0})

print("************************Finish the fine tuning******************************")
# Test trained model
print(sess.run(accuracy, feed_dict={x_ft: mnist.test.images, y_: mnist.test.labels,
                                    ae.keep_prob: 1.0, ae_2nd.keep_prob: 1.0}))
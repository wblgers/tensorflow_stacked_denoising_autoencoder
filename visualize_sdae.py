import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from library.visulize_help import *
import PIL.Image

from library.Autoencoder import Autoencoder

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

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

ae = Autoencoder(n_layers=[n_inputs, n_hidden],
                          transfer_function = tf.nn.relu,
                          optimizer = tf.train.AdamOptimizer(learning_rate = lr),
                          ae_para = [corruption_level, sparse_reg])
ae_2nd = Autoencoder(n_layers=[n_hidden, n_hidden2],
                          transfer_function = tf.nn.relu,
                          optimizer = tf.train.AdamOptimizer(learning_rate = lr),
                          ae_para=[corruption_level, sparse_reg])

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

print("************************First AE******************************")
# Show the features learned by the autoencoder
n = 10
img_size = 28
canvas_features = np.empty((img_size * n, img_size * n))

features = ae.weights['encode'][0]['w'].eval(session=sess).T
image = PIL.Image.fromarray(tile_raster_images(
        X=features,
        img_shape=(img_size, img_size), tile_shape=(n, n),
        tile_spacing=(1, 1)))
plt.figure()
plt.imshow(image, origin="upper", cmap="gray")
plt.show()
image.save('visulize_ae1.png')

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
print("*************************2nd AE*****************************")
# Show the features learned by the autoencoder
n = 10
img_size = 20
canvas_features = np.empty((img_size * n, img_size * n))

features = ae_2nd.weights['encode'][0]['w'].eval(session=sess).T
image = PIL.Image.fromarray(tile_raster_images(
        X=features,
        img_shape=(img_size, img_size), tile_shape=(n, n),
        tile_spacing=(1, 1)))
plt.figure()
plt.imshow(image, origin="upper", cmap="gray")
plt.show()
image.save('visulize_ae2.png')
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

corruption_level = 0.0
sparse_reg = 2

#
n_inputs = 784
n_hidden = 1000
n_outputs = 10

ae = Autoencoder(n_layers=[n_inputs, n_hidden],
                          transfer_function = tf.nn.sigmoid,
                          optimizer = tf.train.AdamOptimizer(learning_rate = 0.001),ae_para = [corruption_level, sparse_reg])
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(n_samples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs,_ = mnist.train.next_batch(batch_size)

        # Fit training using batch data
        temp = ae.partial_fit()
        cost, opt = sess.run(temp, feed_dict={ae.x: batch_xs, ae.keep_prob : ae.in_keep_prob})

        # Compute average loss
        avg_cost += cost / n_samples * batch_size

    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%d,' % (epoch + 1),
              "Cost:", "{:.9f}".format(avg_cost))
ae_test_cost = sess.run(ae.calc_total_cost(), feed_dict={ae.x: mnist.test.images, ae.keep_prob : ae.in_keep_prob})
print("Total cost: " + str(ae_test_cost))

# Show the features learned by the autoencoder
n = 10
img_size = 28
canvas_features = np.empty((img_size * n, img_size * n))

features = ae.weights['encode'][0]['w'].eval(session=sess).T
image = PIL.Image.fromarray(tile_raster_images(
        X=features,
        img_shape=(28, 28), tile_shape=(n, n),
        tile_spacing=(1, 1)))
plt.figure()
plt.imshow(image, origin="upper", cmap="gray")
plt.show()
image.save('visulize.png')

image_compare = np.empty((img_size * 2, img_size * n))
#
count = 0
batch_xs, _ = mnist.test.next_batch(n)
batch_xs += np.random.normal(loc=0, scale=0.3, size=batch_xs.shape)

for j in range(n):
    recover_xs = sess.run(ae.reconstruct(), feed_dict={ae.x: batch_xs, ae.keep_prob : ae.in_keep_prob})
    image_compare[0:img_size, j * img_size:(j + 1) * img_size] = batch_xs[j].reshape([img_size, img_size])
    image_compare[img_size:2 * img_size, j * img_size:(j + 1) * img_size] = recover_xs[j].reshape([img_size, img_size])

# print(image_compare.max())
# print(image_compare.min())
plt.figure()
plt.imshow(image_compare, origin="upper", cmap="gray")
plt.axis('off')
plt.show()

image_compare = 255*scale_to_unit_interval(image_compare)
image_compare = image_compare.astype(np.uint8)
# print(image_compare.max())
# print(image_compare.min())
image = PIL.Image.fromarray(image_compare)
image.save('compare.png')
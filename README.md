# tensorflow_stacked_denoising_autoencoder

## 0. Setup Environment
To run the script, at least following required packages should be satisfied:
- Python 3.5.2
- Tensorflow 1.6.0
- NumPy 1.14.1

You can use Anaconda to install these required packages. For tensorflow, use the following command to make a quick installation under windows:
```
pip install tensorflow
```
## 1. Content
In this project, there are implementations for various kinds of autoencoders. The base python class is library/Autoencoder.py, you can set the value of "ae_para" in the construction function of Autoencoder to appoint corresponding autoencoder.

- ae_para[0]: The corruption level for the input of autoencoder. If ae_para[0]>0, it's a denoising autoencoder;
- aw_para[1]: The coeff for sparse regularization. If ae_para[1]>0, it's a sparse autoencoder.
#### 1.1 autoencoder
Follow the code sample below to construct a autoencoder:
```
corruption_level = 0
sparse_reg = 0

#
n_inputs = 784
n_hidden = 400
n_outputs = 10
lr = 0.001

# define the autoencoder
ae = Autoencoder(n_layers=[n_inputs, n_hidden],
                          transfer_function = tf.nn.relu,
                          optimizer = tf.train.AdamOptimizer(learning_rate = lr),
                          ae_para = [corruption_level, sparse_reg])
```
To visualize the extracted features and  images, check the code in visualize_ae.py.reconstructed
- Extracted features on MNIST:

![Alt text](https://github.com/wblgers/tensorflow_stacked_denoising_autoencoder/raw/master/pjt_images/ae_features.png)
- Reconstructed noisy images after input->encoder->decoder pipeline:

![Alt text](https://github.com/wblgers/tensorflow_stacked_denoising_autoencoder/raw/master/pjt_images/recover_image_ae.png)
#### 1.2 denoising autoencoder
Follow the code sample below to construct a denoising autoencoder:
```
corruption_level = 0.3
sparse_reg = 0

#
n_inputs = 784
n_hidden = 400
n_outputs = 10
lr = 0.001

# define the autoencoder
ae = Autoencoder(n_layers=[n_inputs, n_hidden],
                          transfer_function = tf.nn.relu,
                          optimizer = tf.train.AdamOptimizer(learning_rate = lr),
                          ae_para = [corruption_level, sparse_reg])
```

Test results:
- Extracted features on MNIST:

![Alt text](https://github.com/wblgers/tensorflow_stacked_denoising_autoencoder/raw/master/pjt_images/dae_features.png)
- Reconstructed noisy images after input->encoder->decoder pipeline:

![Alt text](https://github.com/wblgers/tensorflow_stacked_denoising_autoencoder/raw/master/pjt_images/recover_image_dae.png)
#### 1.3 sparse autoencoder
Follow the code sample below to construct a sparse autoencoder:
```
corruption_level = 0
sparse_reg = 2

#
n_inputs = 784
n_hidden = 400
n_outputs = 10
lr = 0.001

# define the autoencoder
ae = Autoencoder(n_layers=[n_inputs, n_hidden],
                          transfer_function = tf.nn.relu,
                          optimizer = tf.train.AdamOptimizer(learning_rate = lr),
                          ae_para = [corruption_level, sparse_reg])
```
#### 1.4 stacked (denoising) autoencoder
For stacked autoencoder, there are more than one autoencoder in this network, in the script of "SAE_Softmax_MNIST.py", I defined two autoencoders:
```
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
```
For the training of SAE on the task of MNIST classification, there are four sequential parts:
1. Training of the first autoencoder;
2. Training of the second autoencoder, based on the output of first ae;
3. Training on the output layer, normally softmax layer, based on the sequential output of first and second ae;
4. Fine-tune on the whole network.

Detailed code can be found in the script "SAE_Softmax_MNIST.py"
## 2. Reference

Class "autoencoder" are based on the tensorflow official models:
https://github.com/tensorflow/models/tree/master/research/autoencoder/autoencoder_models

For the theory on autoencoder, sparse autoencoder, please refer to:
http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/

## 3. My blog for this project
[漫谈autoencoder：降噪自编码器/稀疏自编码器/栈式自编码器（含tensorflow实现）](https://blog.csdn.net/wblgers1234/article/details/81545079)
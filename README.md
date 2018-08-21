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
#### 1.4 stacked (denoising) autoencoder

## 2. Reference


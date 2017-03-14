This is the implementation of our NIPS 2016 paper:

M. Sajjadi, M. Javanmardi, and T. Tasdizen. *Regularization With Stochastic Transformations and Perturbations for Deep Semi-Supervised Learning.* Advances in Neural Information Processing Systems. 2016.

This implementation of *mutual-exclusivity* and *transformation/stability* loss functions is based on the *Caffe*:

http://caffe.berkeleyvision.org/  
https://github.com/BVLC/caffe  

The Installation process of our code is exactly the same as *Caffe*.

The experiments reported in the paper were performed using *cuda-convnet* and *SparseConvNet* frameworks. Here, we provide the Caffe implementation and the scripts for training a semi-supervised MNIST model. 

We added the following layers to Caffe:

- **DataUnlabeled**
- **ImageDataUnlabeled**
- **LossMX**
- **LossTS**

**DataUnlabeled** and **ImageDataUnlabeled** are similar to **Data** and **ImageData** layers respectively. But, they provide unlabeled data that is feeded to unsupervised loss functions. **DataUnlabeled** loads the data from lmdb, leveldb datasets and **ImageDataUnlabeled** loads data from images. Both layers, repeat each sample **nt** times. In case of transformations, these layers provide **nt** randomly transformed versions of each sample that will be feeded to unsupervised loss functions. Both of these layers require the following parameter:

- **nt**: number of times each unlabeled sample should be repeated

**LossMX** is the implementation of the *mutual-exclusivity* loss function. Please refer to the paper for more details. This layer requires the following parameter:

- **lambda**: weight for mutual-exclusivity loss function

**LossTS** is the implementation of the *transformation/stability* loss function. Please refer to the paper for more details. This layer requires the following parameters:

- **lambda**: weight for transformation/stability loss function
- **nt**: number of times each unlabeled sample should be repeated

Please review the following model to see how these layers can be used in practice:

```
$CAFFE_ROOT/examples/mnist/lenet_train_test_ssl.prototxt
```

You need to run the following commands to train semi-supervised MNIST:

```
$CAFFE_ROOT/data/mnist/get_mnist.sh
$CAFFE_ROOT/examples/mnist/create_mnist_ssl.sh
$CAFFE_ROOT/examples/mnist/train_lenet_ssl.sh
```

*create_mnist_ssl.sh* creates two lmdb datasets. The first one is the labeled set and only contains 100 samples (10 per class). The other one is the unlabled set and contains all 60000 samples. The network structure is defined in:

```
$CAFFE_ROOT/examples/mnist/lenet_train_test_ssl.prototxt
```

In this example, we randomly crop each repetition of an unlabeled sample. Then, we try to minimize their difference in predictions using transformation/stability loss function. We also apply the mutual-exclusivity loss function to all the unlabeled data. This increases the accuracy from ~85% to more than 92%.

If you you already have the Caffe installed, you can add the following files to your existing Caffe:

```
$CAFFE_ROOT/src/caffe/layers/loss_mx_layer.cpp
$CAFFE_ROOT/src/caffe/layers/loss_mx_layer.cu
$CAFFE_ROOT/src/caffe/layers/loss_ts_layer.cpp
$CAFFE_ROOT/src/caffe/layers/loss_ts_layer.cu
$CAFFE_ROOT/include/caffe/layers/loss_mx_layer.hpp
$CAFFE_ROOT/include/caffe/layers/loss_ts_layer.hpp
$CAFFE_ROOT/src/caffe/layers/data_unlabeled_layer.cpp
$CAFFE_ROOT/src/caffe/layers/image_data_unlabeled_layer.cpp
$CAFFE_ROOT/include/caffe/layers/data_unlabeled_layer.hpp
$CAFFE_ROOT/include/caffe/layers/image_data_unlabeled_layer.hpp
```

You also need to add **nt** and **lambda** parameters to *caffe.proto*:

```
$CAFFE_ROOT/src/caffe/proto/caffe.proto
```

These are the files needed to run the MNIST example:

```
$CAFFE_ROOT/examples/mnist/convert_mnist_data_100.cpp
$CAFFE_ROOT/examples/mnist/create_mnist_ssl.sh
$CAFFE_ROOT/examples/mnist/train_lenet_ssl.sh
$CAFFE_ROOT/examples/mnist/lenet_solver_ssl.prototxt
$CAFFE_ROOT/examples/mnist/lenet_train_test_ssl.prototxt
```

If your labeled set is small, the data layer constantly prints:

>Restarting data prefetching from start.

In that case you can use the modified data layer file which does not print that message:

```
$CAFFE_ROOT/src/caffe/layers/data_layer.cpp
```

If you have any questions, contact Mehdi Sajjadi at:
mehdi@sci.utah.edu
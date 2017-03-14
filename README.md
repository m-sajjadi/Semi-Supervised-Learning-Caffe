This is the implementation of our NIPS 2016 paper:

M. Sajjadi, M. Javanmardi, and T. Tasdizen. *Regularization With Stochastic Transformations and Perturbations for Deep Semi-Supervised Learning.* Advances in Neural Information Processing Systems. 2016.

This implementation of *mutual-exclusivity* and *transformation/stability* loss functions are based on the *Caffe*:

http://caffe.berkeleyvision.org/  
https://github.com/BVLC/caffe  

The Installation process of our code is exactly the same as *Caffe*.

Here we have provided the code for experiments on CIFAR10, CIFAR100, MNIST, SVHN and NORB. 

For these datasets, first you need to prepare the datasets according to 'README.md' files in the 'Data' folder. Then you need to run the following commands:

```
make mnist
make cifar10
make cifar100
make svhn
make norb
```

Hyper-parameters of these experiments can be modified in 'mnist.cpp', 'cifar10.cpp', 'cifar100.cpp', 'svhn.cpp' and 'norb.cpp'. Improtant hyper-parameters can be set in the begining of each file.

- **nt**: number of times each unlabeled sample should be repeated
- **lambda**: lamda for transformation/stability loss function

If you you already have the Caffe installed, you can add the following files to your existing Caffe:

```
caffe-master/src/caffe/layers/loss_mx_layer.cpp
caffe-master/src/caffe/layers/loss_mx_layer.cu
caffe-master/src/caffe/layers/loss_ts_layer.cpp
caffe-master/src/caffe/layers/loss_ts_layer.cu
caffe-master/include/caffe/layers/loss_mx_layer.hpp
caffe-master/include/caffe/layers/loss_ts_layer.hpp
caffe-master/src/caffe/layers/data_unlabeled_layer.cpp
caffe-master/src/caffe/layers/image_data_unlabeled_layer.cpp
caffe-master/include/caffe/layers/data_unlabeled_layer.hpp
caffe-master/include/caffe/layers/image_data_unlabeled_layer.hpp
```

You also need to add **nt** and **lambda** parameters to *caffe.proto*:

```
caffe-master/src/caffe/proto/caffe.proto
```

These are the files needed to run the MNIST example:

```
caffe-master/examples/mnist/convert_mnist_data_100.cpp
caffe-master/examples/mnist/create_mnist_ssl.sh
caffe-master/examples/mnist/train_lenet_ssl.sh
caffe-master/examples/mnist/lenet_solver_ssl.prototxt
caffe-master/examples/mnist/lenet_train_test_ssl.prototxt
```

If your labeled set is small, the data layer constantly prints:

>Restarting data prefetching from start.

In that case you can use the modified data layer file:

```
caffe-master/src/caffe/layers/data_layer.cpp
```

If you have any questions, contact Mehdi Sajjadi at:
mehdi@sci.utah.edu

This folder contains the implementation of **image classification** task.

In order to use the code, please follow these steps:

## 1- Install requirements

pip install -r requirements.txt


## 2- MNIST, FashionMNIST, CIFAR-10:
To reproduce the results for MNIST using AdaSig clipping, run the follwoing: 

`python myrun.py -myseed 0 -clip_thre 0.1 -learning_rate 0.05 -alpha_learning_rate 0.01 -momentum 0.9 \
-initial_alpha 5 -batch_size 512 -num_epoch 40 \
-delta 1e-5 -epsilon 3.0 -grad_noise_mul_coeff 1.01 \
-dataset 'mnist' -method 'adasig'`

Note that:

You can replace 'mnist' with 'fmnist' or 'cifar10' to generate the reusults for other datasets.


## 3- ImageNette:

To reproduce the results for ImageNette using AdaSig clipping, run the follwoing: 

`python myrun.py -myseed 0 -clip_thre 1.5 -learning_rate 0.001 -alpha_learning_rate 0.02 -momentum 0.9 \
-initial_alpha 5 -batch_size 1024 -num_epoch 50 \
-delta 1e-4 -epsilon 8.0 -grad_noise_mul_coeff 1.01 \
-dataset 'imagenette' -method 'adasig'`


## 4- CelebA:

To reproduce the results for multil-label classification on CelebA using AdaSig clipping, run the follwoing: 

`python myrun.py -myseed ${MS} -clip_thre 0.1 -learning_rate 5e-4 -alpha_learning_rate 0.001 -momentum 0.9 \
-initial_alpha 1 -batch_size 512 -num_epoch 10 \
-delta 5e-6 -epsilon 8.0 -grad_noise_mul_coeff 1.01 \
-dataset 'multilabel-celeba' -method 'adasig'`

Note that:

For smiling label classification: replace 'multilabel-celeba' with 'smiling-celeba'.

## Key Notes:

1- To generate results for different baselines, set the -method argument to either 'vanilla', 'autos', or 'psac'.

2- The random seed for privacy noise generation and batch sampling is set by -myseed argument.

3- The value of the -clip_thre argument only affects the baselines (vanilla, autos, psac), whereas the performance of AdaSig is independent of this value.

4- Other arguments should be set based on best hyper-parameters reported for each method.




Thank you for your attention!

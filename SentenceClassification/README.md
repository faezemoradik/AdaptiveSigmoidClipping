
This folder contains the implementation of **sentence classification** task.

In order to use the code, please follow these steps:

## 1- Install requirements

pip install -r requirements.txt


## 2- SST-2:
In order to fine-tune RoBERTa-base on SST-2 using AdaSig clipping, run the following:

`python myrun.py -myseed 0 -clip_thre 0.1 -learning_rate 5e-4 -alpha_learning_rate 0.005 \
-initial_alpha 1 -batch_size 1000 -num_epoch 3 \
-epsilon 3.0 -grad_noise_mul_coeff 1.01 \
-dataset 'sst2' -method 'adasig'`


## 3- QNLI:
In order to fine-tune RoBERTa-base on QNLI using AdaSig clipping, run the following

`python myrun.py -myseed 0 -clip_thre 0.1 -learning_rate 5e-4 -alpha_learning_rate 0.01 \
-initial_alpha 1 -batch_size 2000 -num_epoch 6 \
-epsilon 3.0 -grad_noise_mul_coeff 1.01 \
-dataset 'qnli' -method 'adasig'`



## Key Notes:

1- To generate results for different baselines, set the -method argument to either 'vanilla', 'autos', or 'psac'.

2- The random seed for privacy noise generation and batch sampling is set by -myseed argument.

3- The value of the -clip_thre argument only affects the baselines (vanilla, autos, psac), whereas the performance of AdaSig is independent of this value.

4- Other arguments should be set based on best hyper-parameters reported for each method and dataset.


Thank you for your attention!

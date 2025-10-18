

from mymain import main
import torch
import random
import numpy as np
import argparse
import os


#----------------------------------------------------------------
def get_parameter():
  parser=argparse.ArgumentParser()
  parser.add_argument("-myseed",default=0,type=int,help="seed")
  parser.add_argument("-clip_thre",default=0.1,type=float,help="Clipping Threshold")
  parser.add_argument("-learning_rate",default=4.0,type=float,help="Learning Rate")
  parser.add_argument("-alpha_learning_rate",default=0.01,type=float,help="Learning Rate for Updating Alpha")
  parser.add_argument("-initial_alpha",default=1.0,type=float,help="Initial Alpha")
  parser.add_argument("-momentum",default=0.9,type=float,help="Momentum")
  parser.add_argument("-batch_size",default=1024,type=int,help=" Batch Size")
  parser.add_argument("-num_epoch",default=40,type=int,help=" Num of Epochs")
  parser.add_argument("-epsilon",default=3.0,type=float,help="Epsilon")
  parser.add_argument("-grad_noise_mul_coeff",default=1.01,type=float,help="Gradient Noise Multiplier")
  parser.add_argument("-dataset",default='mnist',type=str,help="Dataset name")
  parser.add_argument("-method",default='vanilla',type=str,help="Method")
  args= parser.parse_args()
  return args
#--------------------------------------------------------------
args = get_parameter()
myseed = args.myseed
clip_thre= args.clip_thre
learning_rate= args.learning_rate
alpha_learning_rate = args.alpha_learning_rate
initial_alpha = args.initial_alpha
momentum = args.momentum
batch_size= args.batch_size
num_epoch = args.num_epoch
epsilon = args.epsilon
grad_noise_mul_coeff = args.grad_noise_mul_coeff
dataset = args.dataset
method = args.method

print('myseed: ', myseed)
print('clip_thre: ', clip_thre)
print('actual_learning_rate: ', learning_rate)
print('alpha_learning_rate: ', alpha_learning_rate)
print('initial_alpha: ', initial_alpha)
print('momentum: ', momentum)
print('batch_size: ', batch_size)
print('num_epoch: ', num_epoch)
print('epsilon: ', epsilon)
print('grad_noise_mul_coeff: ', grad_noise_mul_coeff)
print('dataset: ', dataset)
print('method: ', method)

torch.backends.cudnn.deterministic=True # to get the reproducible results
# torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(0)
random.seed(myseed) 
torch.manual_seed(myseed) 
np.random.seed(myseed)
torch.random.manual_seed(myseed)


if method == 'adasig':
    logdir = f"logs/{dataset}/{method}/seed{myseed}/lr{learning_rate}_bs{batch_size}_ne{num_epoch}_alr{alpha_learning_rate}_inialp{initial_alpha}_gnmc{grad_noise_mul_coeff}"
else:
    logdir = f"logs/{dataset}/{method}/seed{myseed}/lr{learning_rate}_bs{batch_size}_ne{num_epoch}"

main(dataset=dataset,
    max_grad_norm=clip_thre, grad_noise_mul_coeff= grad_noise_mul_coeff, alpha_learning_rate= alpha_learning_rate, 
    initial_alpha=initial_alpha, batch_size=batch_size, lr=learning_rate, target_epsilon = epsilon, 
    logdir=logdir, target_epoch=num_epoch, method = method )

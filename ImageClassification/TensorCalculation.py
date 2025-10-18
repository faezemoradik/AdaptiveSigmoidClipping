import numpy as np
import torch
import torch.nn.functional as F
import sys
import math


#----------------------------------------------------------
def l2_norm_calculator(u):
  return np.linalg.norm(torch.nn.utils.parameters_to_vector(u).cpu().numpy())
#----------------------------------------------------------
def WeightedAddFunc(a, b, s1, s2):
  add=[]
  for j in range(len(a)):
    add.append(s1*a[j]+s2*b[j])
  return add
#----------------------------------------------------------------------
def Update_model(model, update_params, optimizer, batch_size):
  model.train()
  i=0
  for p in model.parameters():
    p.grad = update_params[i]/batch_size
    i +=1
  optimizer.step()
  return model
#----------------------------------------------------------
def ApplyingPerturbation(update, noise_multiplier, device, rat, clip_thre = 0.1):
  noisy_update=[]
  noise_list=[]
  for j in range(len(update)):
    std= clip_thre*noise_multiplier*torch.ones(size=update[j].size()).to(device)
    mean= torch.zeros(size= update[j].size()).to(device)
    noise= torch.normal(mean, std)
    noisy_update.append(update[j]+ noise)
    noise_list.append(noise)

  if rat== True:
    signal_norm = l2_norm_calculator(update)
    noise_norm = l2_norm_calculator(noise_list)
    ratio = signal_norm/noise_norm
  else:
    ratio = 0

  return noisy_update, ratio
#-----------------------------------------------------------------
def Grad_on_Samples(loss_func, model, optimizer, mini_batch_size, data, target, device):
  model.train()
  output = model(data)
  if loss_func== 'CE':
    loss = F.cross_entropy(output, target)
  elif loss_func== 'BCE':
    loss = F.binary_cross_entropy(output, target)   
  model.zero_grad()
  optimizer.zero_grad()
  loss.backward()

  grads_norm_array = np.zeros(mini_batch_size)
  grads_dicts= dict()
  for i in range(mini_batch_size):
    grads_dicts.update({str(i): []})
    for p in model.parameters():
      grads_dicts[str(i)].append(p.grad_sample[i].detach())

    grads_norm_array[i] = l2_norm_calculator(grads_dicts[str(i)])

  mini_batch_avg_grad = [torch.zeros_like(p).to(device) for p in grads_dicts[str(0)]]
  for i in range(mini_batch_size):
    mini_batch_avg_grad = WeightedAddFunc(mini_batch_avg_grad, grads_dicts[str(i)], 1.0, 1.0/mini_batch_size)


  return grads_dicts, grads_norm_array, output, mini_batch_avg_grad

#----------------------------------------------------------
def dot_product(a, b):
  a_vec= torch.nn.utils.parameters_to_vector(a).cpu().numpy()
  b_vec= torch.nn.utils.parameters_to_vector(b).cpu().numpy()
  dot_prod =  a_vec@b_vec
  
  return dot_prod
#----------------------------------------------------------
def AngleCalculator(a,b):
  a_vec= torch.nn.utils.parameters_to_vector(a).cpu().numpy()
  b_vec= torch.nn.utils.parameters_to_vector(b).cpu().numpy()
  dot_prod =  a_vec@b_vec

  a_norm = l2_norm_calculator(a)
  b_norm = l2_norm_calculator(b)
  angle = 180*np.arccos(dot_prod/(a_norm*b_norm))/math.pi

  return angle
#-------------------------------------------------------
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
def Update_model(model, update_params, optimizer, lr_scheduler, batch_size):
  model.train()
  i=0
  for p in model.parameters():
    p.grad = update_params[i]/batch_size
    i +=1
  optimizer.step()
  lr_scheduler.step()
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
def Grad_on_Samples(model, optimizer, pair_token_ids, mask_ids, labels, device):
  model.train()

  model.zero_grad()
  optimizer.zero_grad()
  batch_loss, prediction = model(pair_token_ids, attention_mask=mask_ids,labels=labels).values()
  batch_correct = (torch.log_softmax(prediction, dim=1).argmax(dim=1) == labels).sum().float()
  batch_loss.backward()

  grads_norm_array = np.zeros(len(labels))
  grads_dicts= dict()
  for i in range(len(labels)):
    grads_dicts.update({str(i): []})
    for p in model.parameters():
      grads_dicts[str(i)].append(p.grad_sample[i].detach())

    grads_norm_array[i] = l2_norm_calculator(grads_dicts[str(i)])

  model.zero_grad()
  optimizer.zero_grad()

  mini_batch_avg_grad = [torch.zeros_like(p).to(device) for p in grads_dicts[str(0)]]
  for i in range(len(labels)):
    mini_batch_avg_grad = WeightedAddFunc(mini_batch_avg_grad, grads_dicts[str(i)], 1.0, 1.0)

  return grads_dicts, grads_norm_array, mini_batch_avg_grad, batch_loss.item(), batch_correct.item()

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
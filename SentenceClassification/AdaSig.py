import numpy as np
import torch
from TensorCalculation import WeightedAddFunc, dot_product, ApplyingPerturbation, Update_model

#------------------------------------------------------------
def AlphaSensitivityCalculation(alpha):
  sensitivity= 0.448/alpha
  return sensitivity
#-------------------------------------------------------------
def SDerivative(grads_norm_array, grads_dicts, summed_s_derivative, alpha = 2.0):

  multipliers = 2*np.exp(-1*alpha*grads_norm_array)/((1+np.exp(-1*alpha*grads_norm_array))**2)
  for i in range(len(multipliers)):
    summed_s_derivative = WeightedAddFunc(grads_dicts[str(i)], summed_s_derivative, multipliers[i], 1.0)

  return summed_s_derivative
#-----------------------------------------------------------------
def MultiplicativeAlphaUpdate(alpha, alpha_learning_rate, noisy_grad_update, noisy_s_derivative, previous_summed_s_derivative, summed_clipped_grad):
  g1= dot_product(summed_clipped_grad, previous_summed_s_derivative)
  g= dot_product(noisy_grad_update, noisy_s_derivative)
  new_alpha = alpha* np.exp(alpha_learning_rate* np.sign(g))
  return new_alpha, g1, g
#-------------------------------------------------------------------
def MySigmoid(x, alpha):
  return 2/(1+np.exp(-1*alpha*x))-1
#-------------------------------------------------------------------
def Sigmoid_Clipping(grads_dicts, grads_norm_array, summed_clipped_grad, alpha = 2.0):
  grad_multipliers = (1/(grads_norm_array+1e-6))*MySigmoid(grads_norm_array, alpha)
  for i in range(len(grads_norm_array)):
    summed_clipped_grad = WeightedAddFunc(grads_dicts[str(i)], summed_clipped_grad, grad_multipliers[i], 1.0)

  return summed_clipped_grad
#--------------------------------------------------------------------
#--------------------------------------------------------------------
def AdaSig(model, optimizer, lr_scheduler, batch_size, learning_rate, summed_s_derivative, previous_summed_s_derivative,
                            summed_clipped_grad, r_noise_multiplier, 
                            grad_noise_multiplier, device, alpha, alpha_learning_rate, 
                            previous_noisy_summed_s_derivative = None):
  
  noisy_summed_grads, ratio= ApplyingPerturbation(summed_clipped_grad, grad_noise_multiplier, 
                                                             device, True, clip_thre = 1.0)

  new_model = Update_model(model, noisy_summed_grads, optimizer, lr_scheduler, batch_size)
  if previous_noisy_summed_s_derivative is not None:
    new_alpha, idl_prod, n_prod = MultiplicativeAlphaUpdate(alpha, alpha_learning_rate, 
                                      noisy_summed_grads, previous_noisy_summed_s_derivative, previous_summed_s_derivative, summed_clipped_grad)

  else:
    new_alpha = 1*alpha
    idl_prod = 0
    n_prod = 0

  r_sensitivity =  AlphaSensitivityCalculation(alpha)
  noisy_summed_s_derivative, _ = ApplyingPerturbation(summed_s_derivative, r_noise_multiplier, 
                                                                device, False, clip_thre = r_sensitivity)

  return new_model, new_alpha, noisy_summed_s_derivative, summed_s_derivative, idl_prod, n_prod, ratio
#--------------------------------------------------------------------------
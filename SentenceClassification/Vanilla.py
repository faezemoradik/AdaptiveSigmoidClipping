import numpy as np
from TensorCalculation import WeightedAddFunc


#----------------------------------------------------------
def Vanilla_Clipping(grads_dicts, grads_norm_array, clip_thre, summed_clipped_grad):
  grad_multipliers= np.zeros(len(grads_norm_array))
  for i in range(len(grads_norm_array)):
    grad_multipliers[i] = min(1.0, clip_thre/grads_norm_array[i])
    summed_clipped_grad = WeightedAddFunc(grads_dicts[str(i)], summed_clipped_grad, grad_multipliers[i], 1.0)

  return summed_clipped_grad
#----------------------------------------------------------
import numpy as np
from TensorCalculation import WeightedAddFunc


#--------------------------------------------------------------
def PSAC_Clipping(grads_dicts, grads_norm_array, clip_thre, summed_clipped_grad, r=0.1):
  grad_multipliers= np.zeros(len(grads_norm_array))
  for i in range(len(grads_norm_array)):
    grad_multipliers[i] =  clip_thre/( grads_norm_array[i]+ r/(grads_norm_array[i]+r) )
    summed_clipped_grad = WeightedAddFunc(grads_dicts[str(i)], summed_clipped_grad, grad_multipliers[i], 1.0)

  return summed_clipped_grad
#--------------------------------------------------------------
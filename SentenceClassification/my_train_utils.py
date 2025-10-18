import torch
from TensorCalculation import ApplyingPerturbation, Update_model, Grad_on_Samples, WeightedAddFunc, AngleCalculator, l2_norm_calculator
from SentenceClassification.Vanilla import Vanilla_Clipping
from AutoS import AutoS_Clipping
from PSAC import PSAC_Clipping
from SentenceClassification.AdaSig import Sigmoid_Clipping, SDerivative, AdaSig
import numpy as np
from data import get_device


def train(model, train_loader, lr_scheduler, optimizer, alpha, pre_noisy_summed_s_derivative, pre_summed_s_derivative, 
          r_noise_multiplier, grad_noise_multiplier, learning_rate, alpha_learning_rate,
          n_acc_steps=1, batch_size=1000, noise_multiplier=1.5, clip_thre=0.1, method='vanilla' ):
    
    device = get_device()
    num_examples = 0
    train_acc = 0
    train_loss = 0

    rem = len(train_loader) % n_acc_steps
    num_batches = len(train_loader)
    num_batches -= rem
    print('n_acc_steps', n_acc_steps)

    bs = train_loader.batch_size if train_loader.batch_size is not None else train_loader.batch_sampler.batch_size
    print(f"training on {num_batches} batches of size {bs}")

    angle_list = []
    gradsnorm_mean_list = []
    gradsnorm_std_list = []
    max_gradsnorm_list =[]
    alpha_list = []
    idl_prod_list=[]
    nidl_prod_list=[]
    ratio_list=[]

    for batch_idx, (pair_token_ids, mask_ids, labels) in enumerate(train_loader):

        if batch_idx > num_batches - 1:
            break

        pair_token_ids, mask_ids, labels = pair_token_ids.to(device), mask_ids.to(device), labels.to(device)
        grads_dicts, grads_norm_array, mini_batch_sum_grad, batch_loss, batch_correct = Grad_on_Samples(model, optimizer, pair_token_ids, mask_ids, labels, device)


        if batch_idx==0:
            summed_clipped_grad = [torch.zeros_like(p).to(device) for p in grads_dicts[str(0)]]
            avg_true_grad = [torch.zeros_like(p).to(device) for p in grads_dicts[str(0)]]
            summed_s_derivative = [torch.zeros_like(p).to(device) for p in grads_dicts[str(0)]]
            grads_norm_list=[]

        grads_norm_list.append(grads_norm_array)
        avg_true_grad =  WeightedAddFunc(avg_true_grad, mini_batch_sum_grad, 1.0, 1.0)

        if method == 'vanilla':
            summed_clipped_grad = Vanilla_Clipping(grads_dicts, grads_norm_array, clip_thre, summed_clipped_grad)

        elif method == 'autos':
            summed_clipped_grad = AutoS_Clipping(grads_dicts, grads_norm_array, clip_thre, summed_clipped_grad) 

        elif method == 'psac':
            summed_clipped_grad = PSAC_Clipping(grads_dicts, grads_norm_array, clip_thre, summed_clipped_grad, r=0.1) 

        elif method == 'adasig': 
            summed_clipped_grad = Sigmoid_Clipping(grads_dicts, grads_norm_array, summed_clipped_grad, alpha = alpha) 
            summed_s_derivative = SDerivative(grads_norm_array, grads_dicts, summed_s_derivative, alpha = alpha)


        if ((batch_idx + 1) % n_acc_steps == 0) or ((batch_idx + 1) == len(train_loader)):
            if method == 'adasig':
                model, alpha, pre_noisy_summed_s_derivative, pre_summed_s_derivative, idl_prod, n_prod, ratio = AdaSig(model, optimizer, lr_scheduler, batch_size, learning_rate, 
                                                                                        summed_s_derivative, pre_summed_s_derivative, summed_clipped_grad, 
                                                                                        r_noise_multiplier, grad_noise_multiplier, 
                                                                                        device, alpha, alpha_learning_rate, 
                                                                                        pre_noisy_summed_s_derivative)
                
                summed_s_derivative = [torch.zeros_like(p).to(device) for p in grads_dicts[str(0)]]
                idl_prod_list.append(idl_prod)
                nidl_prod_list.append(n_prod)
            else:
                noisy_summed_grads, ratio= ApplyingPerturbation(summed_clipped_grad, noise_multiplier, device, True , clip_thre= clip_thre)
                model = Update_model(model, noisy_summed_grads, optimizer, lr_scheduler, batch_size)

            angle =  AngleCalculator(summed_clipped_grad, avg_true_grad)
            angle_list.append(angle)
            full_grad_norm_array = np.concatenate(grads_norm_list)
            gradsnorm_mean_list.append( np.mean(full_grad_norm_array) )
            gradsnorm_std_list.append( np.std(full_grad_norm_array) )
            max_gradsnorm_list.append( np.max(full_grad_norm_array) )
            summed_clipped_grad = [torch.zeros_like(p).to(device) for p in grads_dicts[str(0)]]
            avg_true_grad = [torch.zeros_like(p).to(device) for p in grads_dicts[str(0)]]
            grads_norm_list=[]
            alpha_list.append(alpha)
            ratio_list.append(ratio)
            

        train_acc += batch_correct
        train_loss += (batch_loss*len(pair_token_ids))
        num_examples += len(pair_token_ids)

    train_loss /= num_examples
    train_acc /= num_examples

    print(f'Train set: Average loss: {train_loss:.4f}, 'f'Accuracy: ({train_acc:.3f})')

    return train_loss, train_acc, alpha_list, pre_noisy_summed_s_derivative, pre_summed_s_derivative, angle_list, gradsnorm_mean_list, gradsnorm_std_list, max_gradsnorm_list, idl_prod_list, nidl_prod_list, ratio_list


def test(model, test_loader):
    device = get_device()
    model.eval()
    num_examples = 0
    test_loss = 0
    test_acc = 0

    with torch.no_grad():
        for batch_idx, (pair_token_ids, mask_ids, labels) in enumerate(test_loader):
            pair_token_ids, mask_ids, labels = pair_token_ids.to(device), mask_ids.to(device), labels.to(device)
            batch_loss, prediction = model(pair_token_ids, attention_mask=mask_ids, labels=labels).values()
            batch_correct =(torch.log_softmax(prediction, dim=1).argmax(dim=1) == labels).sum().float()

            num_examples += len(pair_token_ids)
            test_loss += (batch_loss.item()*len(pair_token_ids))
            test_acc += batch_correct.item()


    test_loss /= num_examples
    test_acc /= num_examples

    print(f'Test set: Average loss: {test_loss:.4f}, 'f'Accuracy: ({test_acc:.3f})')

    return test_loss, test_acc


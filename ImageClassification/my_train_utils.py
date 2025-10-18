import torch
import torch.nn.functional as F
from TensorCalculation import ApplyingPerturbation, Update_model, Grad_on_Samples, WeightedAddFunc, AngleCalculator, l2_norm_calculator
from ImageClassification.Vanilla import Vanilla_Clipping
from AutoS import AutoS_Clipping
from PSAC import PSAC_Clipping
from ImageClassification.AdaSig import Sigmoid_Clipping, SDerivative, AdaSig
import numpy as np


def get_device(print_stat= False):
    use_cuda = torch.cuda.is_available()
    if print_stat:
        if use_cuda:
            print("CUDA is available")
        else: 
            print("CUDA is not available")
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device


def train(loss_func, model, train_loader, optimizer, alpha, pre_noisy_summed_s_derivative, pre_summed_s_derivative, 
          r_noise_multiplier, grad_noise_multiplier, learning_rate, alpha_learning_rate,
          n_acc_steps=1, batch_size=2048, noise_multiplier=1.5, clip_thre=0.1, method='vanilla' ):
    
    device = get_device()
    num_examples = 0
    correct = 0
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


    for batch_idx, (data, target) in enumerate(train_loader):

        if batch_idx > num_batches - 1:
            break

        data, target = data.to(device), target.to(device)
        grads_dicts, grads_norm_array, output, mini_batch_avg_grad = Grad_on_Samples(loss_func, model, optimizer, bs, data, target, device)


        if batch_idx==0:
            summed_clipped_grad = [torch.zeros_like(p).to(device) for p in grads_dicts[str(0)]]
            avg_true_grad = [torch.zeros_like(p).to(device) for p in grads_dicts[str(0)]]
            summed_s_derivative = [torch.zeros_like(p).to(device) for p in grads_dicts[str(0)]]
            grads_norm_list=[]

        grads_norm_list.append(grads_norm_array)
        avg_true_grad =  WeightedAddFunc(avg_true_grad, mini_batch_avg_grad, 1.0, 1.0/ n_acc_steps)

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
                model, alpha, pre_noisy_summed_s_derivative, pre_summed_s_derivative, idl_prod, n_prod, ratio = AdaSig(model, optimizer, batch_size, learning_rate, 
                                                                                        summed_s_derivative, pre_summed_s_derivative, summed_clipped_grad, 
                                                                                        r_noise_multiplier, grad_noise_multiplier, 
                                                                                        device, alpha, alpha_learning_rate, 
                                                                                        pre_noisy_summed_s_derivative)
                
                summed_s_derivative = [torch.zeros_like(p).to(device) for p in grads_dicts[str(0)]]
                idl_prod_list.append(idl_prod)
                nidl_prod_list.append(n_prod)
            else:
                noisy_summed_grads, ratio = ApplyingPerturbation(summed_clipped_grad, noise_multiplier, device, True, clip_thre= clip_thre)
                model = Update_model(model, noisy_summed_grads, optimizer, batch_size)

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
            


        if loss_func== 'CE':
            train_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            num_examples += len(data)
        elif loss_func== 'BCE':
            train_loss += F.binary_cross_entropy(output, target, reduction='sum').item()
            pred = (output > 0.5).float()
            correct += (pred == target).sum().item()
            num_labels= target.shape[1]
            num_examples += (len(data)*num_labels)


    train_loss /= num_examples
    train_acc = 100. * correct / num_examples

    print(f'Train set: Average loss: {train_loss:.4f}, '
            f'Accuracy: {correct}/{num_examples} ({train_acc:.2f}%)')

    return train_loss, train_acc, alpha_list, pre_noisy_summed_s_derivative, pre_summed_s_derivative, angle_list, gradsnorm_mean_list, gradsnorm_std_list, max_gradsnorm_list, idl_prod_list, nidl_prod_list, ratio_list


def test(loss_func, model, test_loader):
    device = next(model.parameters()).device
    model.eval()
    num_examples = 0
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            if loss_func== 'CE':
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
                num_examples += len(data)
            elif loss_func == 'BCE':
                test_loss += F.binary_cross_entropy(output, target, reduction='sum').item()
                pred = (output > 0.5).float()
                correct += (pred == target).sum().item()
                num_labels= target.shape[1]
                num_examples += (len(data)*num_labels)


    test_loss /= num_examples
    test_acc = 100. * correct / num_examples

    print(f'Test set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{num_examples} ({test_acc:.2f}%)')

    return test_loss, test_acc

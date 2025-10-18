import os
import pickle
from my_train_utils import train, test
from opacus.grad_sample import GradSampleModule
from opacus.accountants.utils import get_noise_multiplier
from opacus.accountants.analysis import rdp as privacy_analysis
import numpy as np
from data import get_num_params, get_device, MNLIDataRoberta, QNLIDataRoberta, QQPDataRoberta, SST2DataRoberta
from transformers import RobertaForSequenceClassification
from transformers.optimization import AdamW, get_linear_schedule_with_warmup


def main(dataset = 'mnli-matched', grad_noise_mul_coeff= 1.01, alpha_learning_rate=0.01, 
         initial_alpha=1.0, batch_size=1000, lr=1, target_epsilon = 3.0, 
         max_grad_norm=0.1, target_epoch=3, logdir=None, method = 'vanilla'):


    device = get_device(print_stat=True)
    mini_batch_size = 20

    if dataset=='mnli-matched':
        model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=3)
        mydata = MNLIDataRoberta('matched')
        train_loader, test_loader = mydata.get_data_loaders(batch_size=mini_batch_size)

    elif dataset=='mnli-mismatched':
        model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=3)
        mydata = MNLIDataRoberta('mismatched')
        train_loader, test_loader = mydata.get_data_loaders(batch_size=mini_batch_size)

    elif dataset=='qnli':
        model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
        mydata = QNLIDataRoberta()
        train_loader, test_loader = mydata.get_data_loaders(batch_size=mini_batch_size)

    elif dataset=='qqp':
        model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
        mydata = QQPDataRoberta()
        train_loader, test_loader = mydata.get_data_loaders(batch_size=mini_batch_size)

    elif dataset=='sst2':
        model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
        mydata = SST2DataRoberta()
        train_loader, test_loader = mydata.get_data_loaders(batch_size=mini_batch_size)


    num_train_samples= mydata.train_size
    print('num_train_samples', num_train_samples)
    print('len train loader: ', len(train_loader))

    optimizer= AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    num_training_steps= target_epoch*int(num_train_samples/batch_size)
    print('num_training_steps: ', num_training_steps)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, 
                                                   num_training_steps=num_training_steps)

    model.to(device)
    print(f"model has {get_num_params(model)} parameters")


    bs = batch_size
    assert bs % mini_batch_size == 0
    n_acc_steps = bs // mini_batch_size


    model = GradSampleModule(model) ## wrap the model so that the per sample gradients are accessible.

    target_delta = 0.5*(1/num_train_samples)
    print('target_delta: ', target_delta)

    noise_multiplier = get_noise_multiplier(target_epsilon= target_epsilon,  target_delta= target_delta,
                                            sample_rate= bs/num_train_samples, epochs = target_epoch)
    print('noise multiplier is: ', noise_multiplier)

    r_noise_multiplier = noise_multiplier/np.sqrt(1-grad_noise_mul_coeff**(-2))
    grad_noise_mul = grad_noise_mul_coeff*noise_multiplier
    print('grad noise multiplier: ', grad_noise_mul)
    print('r_noise_multiplier: ', r_noise_multiplier)
    pre_noisy_summed_s_derivative = None
    pre_summed_s_derivative = None

    test_loss_array = np.zeros(target_epoch+1)
    test_acc_array = np.zeros(target_epoch+1)
    train_loss_array = np.zeros(target_epoch+1)
    train_acc_array = np.zeros(target_epoch+1)
    gradsnorm_mean_list= []
    gradsnorm_std_list= []
    max_gradsnorm_list = []
    angle_list = []
    alpha_list = [initial_alpha]
    idl_prod_list=[]
    nidl_prod_list=[]
    ratio_list=[]

    train_loss_array[0], train_acc_array[0]= test(model, train_loader)
    test_loss_array[0], test_acc_array[0]= test(model, test_loader)

    saved_dict= dict()

    for epoch in range(0, target_epoch):
        print(f"\nEpoch: {epoch}")

        train_loss, train_acc, alpha, pre_noisy_summed_s_derivative, pre_summed_s_derivative, angle, gradsnorm_mean, gradsnorm_std, max_gradsnorm, idl_prod, nidl_prod, ratio = train(model, train_loader, lr_scheduler, optimizer, 
                                                                                                                                                                               alpha_list[-1], pre_noisy_summed_s_derivative, pre_summed_s_derivative,
                                                                                                                                                                                r_noise_multiplier, grad_noise_mul, lr, alpha_learning_rate,
                                                                                                                                                                                n_acc_steps=n_acc_steps, batch_size=bs, 
                                                                                                                                                                                noise_multiplier= noise_multiplier, 
                                                                                                                                                                                clip_thre=max_grad_norm, method= method)
        test_loss, test_acc = test(model, test_loader)

        if noise_multiplier > 0:
            DEFAULT_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
            rdp_sgd = privacy_analysis.compute_rdp(q= bs /num_train_samples, 
                                                   noise_multiplier= noise_multiplier,
                                                   steps= (epoch+1)*num_train_samples// bs,
                                                    orders= DEFAULT_ALPHAS)


            epsilon, _ = privacy_analysis.get_privacy_spent(orders= DEFAULT_ALPHAS, 
                                                            rdp= rdp_sgd, 
                                                            delta = target_delta)


            print(f"epsilon = {epsilon:.3f}")

        train_loss_array[epoch+1]= train_loss
        train_acc_array[epoch+1]= train_acc
        test_loss_array[epoch+1]= test_loss
        test_acc_array[epoch+1]= test_acc
        alpha_list = alpha_list + alpha
        angle_list = angle_list + angle
        gradsnorm_mean_list = gradsnorm_mean_list + gradsnorm_mean
        gradsnorm_std_list = gradsnorm_std_list+ gradsnorm_std
        max_gradsnorm_list = max_gradsnorm_list+ max_gradsnorm
        idl_prod_list = idl_prod_list+ idl_prod
        nidl_prod_list = nidl_prod_list+ nidl_prod
        ratio_list = ratio_list+ ratio
    
    saved_dict['train_loss']= train_loss_array
    saved_dict['train_acc']= train_acc_array
    saved_dict['test_loss']= test_loss_array
    saved_dict['test_acc']= test_acc_array
    saved_dict['alpha']= np.array(alpha_list)
    saved_dict['grad_norm_mean']= np.array(gradsnorm_mean_list)
    saved_dict['grad_norm_std']= np.array(gradsnorm_std_list)
    saved_dict['grad_norm_max']= np.array(max_gradsnorm_list)
    saved_dict['angle']= np.array(angle_list)
    saved_dict['idl_prod']= np.array(idl_prod_list)
    saved_dict['nidl_prod']= np.array(nidl_prod_list)
    saved_dict['ratio']= np.array(ratio_list)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    file_path = os.path.join(logdir, 'saved_dict')
    with open(file_path, 'wb') as file:
        pickle.dump(saved_dict, file)



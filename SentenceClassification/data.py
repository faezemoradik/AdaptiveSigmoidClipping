import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import RobertaTokenizer
from datasets import load_dataset


def get_device(print_stat= False):
    use_cuda = torch.cuda.is_available()
    if print_stat:
        if use_cuda:
            print("CUDA is available")
        else: 
            print("CUDA is not available")
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device
#--------------------------------------------------------------
def get_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
#--------------------------------------------------------------

class MNLIDataRoberta(Dataset):
    def __init__(self, name):
        self.label_dict = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
        self.ds = load_dataset('glue', 'mnli')
        self.train_ds = self.ds['train']
        if name == 'matched':
            self.val_ds = self.ds['validation_matched']
        elif name == 'mismatched':
            self.val_ds = self.ds['validation_mismatched']           
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
        self.init_data()

    def init_data(self):
        self.train_data, self.train_size = self.load_data(self.train_ds)
        self.val_data, self.val_size = self.load_data(self.val_ds)

    def load_data(self, df):
        token_ids = []
        mask_ids = []
        y = []

        premise_list = df['premise']
        hypothesis_list = df['hypothesis']
        label_list = df['label']

        for (premise, hypothesis, label) in zip(premise_list, hypothesis_list, label_list):
            premise_id = self.tokenizer.encode(premise, add_special_tokens=False, truncation=True, max_length=256)
            hypothesis_id = self.tokenizer.encode(hypothesis, add_special_tokens=False, truncation=True, max_length=256)
            pair_token_ids = [self.tokenizer.cls_token_id] + premise_id + [self.tokenizer.sep_token_id] + hypothesis_id + [self.tokenizer.sep_token_id]
            attention_mask_ids = torch.tensor([1] * len(pair_token_ids))

            token_ids.append(torch.tensor(pair_token_ids))
            mask_ids.append(attention_mask_ids)
            y.append(label)

        token_ids = pad_sequence(token_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        mask_ids = pad_sequence(mask_ids, batch_first=True, padding_value=0)
        y = torch.tensor(y)
        dataset = TensorDataset(token_ids, mask_ids, y)
        print(len(dataset))
        return dataset, len(dataset)

    def get_data_loaders(self, batch_size=32):
        train_loader = DataLoader(
            self.train_data,
            shuffle=True,
            batch_size=batch_size,
            pin_memory=True, 
            drop_last=True
        )

        val_loader = DataLoader(
            self.val_data,
            shuffle=False,
            batch_size=batch_size, 
            pin_memory=True
        )

        return train_loader, val_loader
    
#-------------------------------------------------------------------------------
class QNLIDataRoberta(Dataset):
    def __init__(self):
        self.label_dict = {'entailment': 0, 'not_entailment': 1}
        self.ds = load_dataset('glue', 'qnli')
        self.train_ds = self.ds['train']
        self.val_ds = self.ds['validation']

        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
        self.init_data()

    def init_data(self):
        self.train_data, self.train_size = self.load_data(self.train_ds)
        self.val_data, self.val_size = self.load_data(self.val_ds)

    def load_data(self, df):
        token_ids = []
        mask_ids = []
        y = []

        question_list = df['question']
        sentence_list = df['sentence']
        label_list = df['label']

        for (question, sentence, label) in zip(question_list, sentence_list, label_list):
            question_id = self.tokenizer.encode(question, add_special_tokens=False, truncation=True, max_length=256)
            sentence_id = self.tokenizer.encode(sentence, add_special_tokens=False, truncation=True, max_length=256)
            pair_token_ids = [self.tokenizer.cls_token_id] + question_id + [self.tokenizer.sep_token_id] + sentence_id + [self.tokenizer.sep_token_id]
            attention_mask_ids = torch.tensor([1] * len(pair_token_ids))

            token_ids.append(torch.tensor(pair_token_ids))
            mask_ids.append(attention_mask_ids)
            y.append(label)

        token_ids = pad_sequence(token_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        mask_ids = pad_sequence(mask_ids, batch_first=True, padding_value=0)
        y = torch.tensor(y)
        dataset = TensorDataset(token_ids, mask_ids, y)
        print(len(dataset))
        return dataset, len(dataset)

    def get_data_loaders(self, batch_size=32):
        train_loader = DataLoader(
            self.train_data,
            shuffle=True,
            batch_size=batch_size,
            pin_memory=True,
            drop_last=True
        )

        val_loader = DataLoader(
            self.val_data,
            shuffle=False,
            batch_size=batch_size,
            pin_memory=True
        )

        return train_loader, val_loader

#-----------------------------------------------------------------------------
class QQPDataRoberta(Dataset):
    def __init__(self):
        self.label_dict = {'not_duplicate': 0, 'duplicate': 1}
        self.ds = load_dataset('glue', 'qqp')
        self.train_ds = self.ds['train']
        self.val_ds = self.ds['validation']

        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
        self.init_data()

    def init_data(self):
        self.train_data, self.train_size = self.load_data(self.train_ds)
        self.val_data, self.val_size = self.load_data(self.val_ds)

    def load_data(self, df):
        token_ids = []
        mask_ids = []
        y = []

        question1_list = df['question1']
        question2_list = df['question2']
        label_list = df['label']

        for (question1, question2, label) in zip(question1_list, question2_list, label_list):
            question1_id = self.tokenizer.encode(question1, add_special_tokens=False, truncation=True, max_length=256)
            question2_id = self.tokenizer.encode(question2, add_special_tokens=False, truncation=True, max_length=256)
            pair_token_ids = [self.tokenizer.cls_token_id] + question1_id + [self.tokenizer.sep_token_id] + question2_id + [self.tokenizer.sep_token_id]
            attention_mask_ids = torch.tensor([1] * len(pair_token_ids))

            token_ids.append(torch.tensor(pair_token_ids))
            mask_ids.append(attention_mask_ids)
            y.append(label)

        token_ids = pad_sequence(token_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        mask_ids = pad_sequence(mask_ids, batch_first=True, padding_value=0)
        y = torch.tensor(y)
        dataset = TensorDataset(token_ids, mask_ids, y)
        print(len(dataset))
        return dataset, len(dataset)

    def get_data_loaders(self, batch_size=32):
        train_loader = DataLoader(
            self.train_data,
            shuffle=True,
            batch_size=batch_size,
            pin_memory=True, 
            drop_last=True
        )

        val_loader = DataLoader(
            self.val_data,
            shuffle=False,
            batch_size=batch_size, 
            pin_memory=True
        )

        return train_loader, val_loader

#-------------------------------------------------------------------------------------
class SST2DataRoberta(Dataset):
    def __init__(self):
        self.label_dict = {'negative': 0, 'positive': 1}
        self.ds = load_dataset('glue', 'sst2')
        self.train_ds = self.ds['train']
        self.val_ds = self.ds['validation']

        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
        self.init_data()

    def init_data(self):
        self.train_data, self.train_size = self.load_data(self.train_ds)
        self.val_data, self.val_size = self.load_data(self.val_ds)

    def load_data(self, df):
        token_ids = []
        mask_ids = []
        y = []

        sentence_list = df['sentence']
        label_list = df['label']

        for (sentence, label) in zip(sentence_list, label_list):
            sentence_id = self.tokenizer.encode(sentence, add_special_tokens=False, truncation=True, max_length=256)
            pair_token_ids = [self.tokenizer.cls_token_id] + sentence_id + [self.tokenizer.sep_token_id]
            attention_mask_ids = torch.tensor([1] * len(pair_token_ids))

            token_ids.append(torch.tensor(pair_token_ids))
            mask_ids.append(attention_mask_ids)
            y.append(label)

        token_ids = pad_sequence(token_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        mask_ids = pad_sequence(mask_ids, batch_first=True, padding_value=0)
        y = torch.tensor(y)
        dataset = TensorDataset(token_ids, mask_ids, y)
        print(len(dataset))
        return dataset, len(dataset)

    def get_data_loaders(self, batch_size=32):
        train_loader = DataLoader(
            self.train_data,
            shuffle=True,
            batch_size=batch_size,
            pin_memory=True, 
            drop_last=True
        )

        val_loader = DataLoader(
            self.val_data,
            shuffle=False,
            batch_size=batch_size, 
            pin_memory=True
        )

        return train_loader, val_loader
    
#------------------------------------------------------------------------------------




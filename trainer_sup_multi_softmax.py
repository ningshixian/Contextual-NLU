# -*- encoding: utf-8 -*-

import random
import time
from typing import List

import jsonlines
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from scipy.stats import spearmanr
from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch import Tensor,bool,long
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizer, AlbertTokenizer
import pandas as pd
import jieba
import os

from visdom import Visdom

# 基本参数
from amsoftmax import AMsoftmax

EPOCHS = 100
BATCH_SIZE = 64
LR = 2e-5
MAXLEN = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
EARLY=5
# 预训练模型目录
# BERT = '/data/wanghao/work/pretrained_models/chinese_roberta_wwm_ext_pytorch/'
BERT = '/data/wanghao/work/pretrained_models/uer_roberta_word_base/'
WORD_ATT = '../model_pretrain/train_mask/pretrained_model_word_att_high_swarp_epoch_12/'
BERT_WWM_EXT = 'pretrained_model/bert_wwm_ext_pytorch'
ROBERTA = 'pretrained_model/roberta_wwm_ext_pytorch'
train_set = "./dialogue_train_notest_ningNew_nofalse_clean.csv"
model_path = BERT
# tokenizer = BertTokenizer.from_pretrained(model_path,do_lower_case=True,additional_special_tokens=["[PIC]", "[JSON]", "[HTTP]", "[PHONE]", "[SUBPH]"])
tokenizer = AlbertTokenizer.from_pretrained(model_path,do_lower_case=True)
# 微调后参数存放位置
SAVE_PATH = './saved_model/seq128_b64_class_rand_true_notest_uer_ningnew_amsoft_trueloss_nlu_cosent_like_uda_%s_%s_%s_2.pt'
user_dict = ["[PIC]", "[JSON]", "[HTTP]", "[PHONE]", "[SUBPH]"]
tokenizer.add_tokens(user_dict)
print(tokenizer.encode_plus(
    text="[PAD][JSON][PIC]/你好呀 请问中石化充卡多久可以到账呢",
    text_pair=None,
    add_special_tokens=True,
    return_token_type_ids=True
))
# 数据位置



def load_data(name: str, path: str,data_type:str) -> List:
    """根据名字加载不同的数据集
    """
    #TODO: 把lqcmc的数据生成正负样本, 拿来做测试
    def load_snli_data(path):
        data=pd.read_csv(path)
        return [(data["sentence1"][i],data["sentence2"][i],str(ilabel)) for i,ilabel in enumerate(data["label"]) ]
    def label_num(path):
        data=pd.read_csv(path)
        labels=[data["sentence1"][i] for i,ilabel in enumerate(data["label"]) if isinstance(data["sentence1"][i],str) and isinstance(data["sentence2"][i],str)]
        return len(set(labels))

    def load_lqcmc_data(path):
        data=pd.read_csv(path)
        return [(data["sentence1"][i],data["sentence2"][i],str(ilabel)) for i,ilabel in enumerate(data["label"]) if isinstance(data["sentence1"][i],str) and isinstance(data["sentence2"][i],str)]
    def load_sts_data(path):
        data=pd.read_csv(path)
        return [(data["sentence1"][i],data["sentence2"][i],str(ilabel)) for i,ilabel in enumerate(data["label"]) if isinstance(data["sentence1"][i],str) and isinstance(data["sentence2"][i],str)]
    def load_threshold_data(path):
        data=pd.read_csv(path)
        return [(data["sentence1"][i],data["sentence2"][i],str(ilabel if ilabel else -1)) for i,ilabel in enumerate(data["label"]) if isinstance(data["sentence1"][i],str) and isinstance(data["sentence2"][i],str)]
    if name == 'sts':
        return load_sts_data(path)
    if data_type == "label_num":
        return label_num(path)
    if data_type =="train":
        return load_lqcmc_data(path)
    elif data_type == "threshold":
        return load_threshold_data(path)
    else:
        return load_snli_data(path)



class SimTrainDataset(Dataset):
    """训练数据集, 重写__getitem__和__len__方法
    """
    def __init__(self, data: List,class_label:List):
        # self.data = [idata for idata in data if int(idata[2])==1 and isinstance(idata[0],str) and isinstance(idata[1],str)]
        self.data_sim = {}
        for raw in data:
            if int(raw[2])==1 and isinstance(raw[0],str) and isinstance(raw[1],str):
                if raw[0] not in self.data_sim:
                    self.data_sim[raw[0]]=[raw[0],raw[1]]
                else:
                    self.data_sim[raw[0]].extend([raw[1]])
        self.data=list(self.data_sim.keys())
        self.class_label = class_label

    def __len__(self):
        return len(self.data)

    def is_chinese(self,check_str):
        """
        检查整个字符串全是中文
        :param string: 需要检查的字符串
        :return: bool
        """
        for ch in check_str:
            if u'\u4e00' <= ch <= u'\u9fff':
                pass
            else:
                return False
        return True

    def text_2_id(self, text: str):
        return tokenizer.encode_plus(
            text=text,
            text_pair=None,
            add_special_tokens=True,
            return_token_type_ids=True
        )

    def __getitem__(self, index):
        line = self.data[index]
        sim_data=random.sample(self.data_sim[line],2)
        assert len(sim_data[0])>0
        assert len(sim_data[1])>0
        return self.text_2_id(sim_data[0]), self.text_2_id(sim_data[1]), 1,int(self.class_label.index(line))
    
    
class AntiSimTrainDataset(Dataset):
    """测试数据集, 重写__getitem__和__len__方法
    """
    def __init__(self, data: List,class_label: List):
        self.data = list(set([raw[0] for raw in data if isinstance(raw[0],str)]))
        self.sim_data={}
        for raw in data:
            if raw[0] not in self.sim_data:
                self.sim_data[raw[0]]=[raw[0],raw[1]]
            else:
                self.sim_data[raw[0]].extend([raw[1]])
        self.class_label = class_label
    def __len__(self):
        return 500000
    def is_chinese(self,check_str):
        """
        检查整个字符串全是中文
        :param string: 需要检查的字符串
        :return: bool
        """
        for ch in check_str:
            if u'\u4e00' <= ch <= u'\u9fff':
                pass
            else:
                return False
        return True

    def text_2_id(self, text: str):
        return tokenizer.encode_plus(
            text=text,
            text_pair=None,
            add_special_tokens=True,
            return_token_type_ids=True
        )
    
    def __getitem__(self, index):
        line = random.sample(self.data,2)
        anti_1=random.sample(self.sim_data[line[0]],1)
        anti_2=random.sample(self.sim_data[line[1]],1)
        class_label_1=self.class_label.index(line[0])
        class_label_2=self.class_label.index(line[1])
        assert len(anti_1[0])>0
        assert len(anti_2[0])>0
        return self.text_2_id(anti_1[0]), self.text_2_id(anti_2[0]), 0,class_label_1,class_label_2

class TestDataset(Dataset):
    """测试数据集, 重写__getitem__和__len__方法
    """
    def __init__(self, data: List):
        self.data = data
    def __len__(self):
        return len(self.data)

    def is_chinese(self,check_str):
        """
        检查整个字符串全是中文
        :param string: 需要检查的字符串
        :return: bool
        """
        for ch in check_str:
            if u'\u4e00' <= ch <= u'\u9fff':
                pass
            else:
                return False
        return True

    def text_2_id(self, text: str):
        return tokenizer.encode_plus(
            text=text,
            text_pair=None,
            add_special_tokens=True,
            return_token_type_ids=True
        )

    def __getitem__(self, index):
        line = self.data[index]

        return self.text_2_id(line[0]), self.text_2_id(line[1]), int(line[2])

def pad_to_maxlen(input_ids, max_len, pad_value=0):
    if len(input_ids) >= max_len:
        input_ids = input_ids[:max_len]
    else:
        input_ids = input_ids + [pad_value] * (max_len - len(input_ids))
    # print(len(input_ids))
    return input_ids

def collate_fn(batch):
    # 按batch进行padding获取当前batch中最大长度
    max_len=0
    for raw in batch:
        len_1=len(raw[0]["input_ids"])
        len_2=len(raw[1]["input_ids"])
        max_len=max([len_1,len_2,max_len])

    if max_len > MAXLEN:
        max_len = MAXLEN

    # 定一个全局的max_len
    # max_len = 128

    batch_item_num=0
    sentence_1={}
    sentence_2={}
    label=[]
    class_label=[]
    class_label_anti=[]
    for item in batch:
        batch_item_num=len(item)
        if "input_ids" not in sentence_1:
            sentence_1["input_ids"]=[pad_to_maxlen(item[0]['input_ids'], max_len=max_len)]
        else:
            sentence_1["input_ids"].append(pad_to_maxlen(item[0]['input_ids'], max_len=max_len))
        if "attention_mask" not in sentence_1:
            sentence_1["attention_mask"]=[pad_to_maxlen(item[0]['attention_mask'], max_len=max_len)]
        else:
            sentence_1["attention_mask"].append(pad_to_maxlen(item[0]['attention_mask'], max_len=max_len))
        if "token_type_ids" not in sentence_1:
            sentence_1["token_type_ids"]=[pad_to_maxlen(item[0]['token_type_ids'], max_len=max_len)]
        else:
            sentence_1["token_type_ids"].append(pad_to_maxlen(item[0]['token_type_ids'], max_len=max_len))
        if "input_ids" not in sentence_2:
            sentence_2["input_ids"]=[pad_to_maxlen(item[1]['input_ids'], max_len=max_len)]
        else:
            sentence_2["input_ids"].append(pad_to_maxlen(item[1]['input_ids'], max_len=max_len))
        if "attention_mask" not in sentence_2:
            sentence_2["attention_mask"]=[pad_to_maxlen(item[1]['attention_mask'], max_len=max_len)]
        else:
            sentence_2["attention_mask"].append(pad_to_maxlen(item[1]['attention_mask'], max_len=max_len))
        if "token_type_ids" not in sentence_2:
            sentence_2["token_type_ids"]=[pad_to_maxlen(item[1]['token_type_ids'], max_len=max_len)]
        else:
            sentence_2["token_type_ids"].append(pad_to_maxlen(item[1]['token_type_ids'], max_len=max_len))
        label.append(item[2])
        if batch_item_num==4:
            class_label.append(item[3])
        elif batch_item_num==5:
            class_label_anti.append(item[3])
            class_label_anti.append(item[4])
    for raw in sentence_1:
        # print(len(sentence_1[raw]),len(sentence_1[raw][0]))
        sentence_1[raw]=torch.tensor(sentence_1[raw],dtype=torch.long)
    for raw in sentence_2:
        sentence_2[raw]=torch.tensor(sentence_2[raw],dtype=torch.long)
    label=torch.tensor(label,dtype=torch.float)
    if batch_item_num==4:
        class_label=torch.tensor(class_label,dtype=torch.long)
    if batch_item_num==5:
        class_label_anti=torch.tensor(class_label_anti,dtype=torch.long)
    if batch_item_num==4:
        return sentence_1,sentence_2,label,class_label
    elif batch_item_num==5:
        return sentence_1,sentence_2,label,class_label_anti
    else:
        return sentence_1,sentence_2,label

class SimcseModel(nn.Module):
    """Simcse有监督模型定义"""
    def __init__(self, pretrained_model: str, pooling: str ,label_num: int):
        super(SimcseModel, self).__init__()
        # config = BertConfig.from_pretrained(pretrained_model)   # 有监督不需要修改dropout
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.bert.resize_token_embeddings(len(tokenizer))
        self.pooling = pooling
        self.linear= nn.Linear(768,label_num)
        self.amsoftmax=AMsoftmax(0.25,20,self.linear,DEVICE)
    def forward(self, input_ids, attention_mask, token_type_ids,labels):
        
        # out = self.bert(input_ids, attention_mask, token_type_ids)
        out = self.bert(input_ids, attention_mask, token_type_ids)

        if self.pooling == 'cls':
            # return out.last_hidden_state[:, 0],self.linear(out.last_hidden_state[:, 0])  # [batch, 768]
            if labels!=None:
                return out.last_hidden_state[:, 0],self.amsoftmax.loss_logic(out.last_hidden_state[:, 0],labels)  # [batch, 768]
            else:
                return out.last_hidden_state[:, 0],self.linear(out.last_hidden_state[:, 0])  # [batch, 768]
            # return out.mix_cls,out.attentions  # [batch, 768]
        if self.pooling == 'mix_cls':
            # return out.last_hidden_state[:, 0],out.attentions  # [batch, 768]
            return out.mix_cls,out.attentions  # [batch, 768]
        if self.pooling == 'pooler':
            return out.pooler_output,out.attentions        # [batch, 768]
        
        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)    # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1),out.attentions       # [batch, 768]
        
        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)    # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)    # [batch, 768, seqlen]                   
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1) # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)   # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)     # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1),out.attentions     # [batch, 768]
                  
            
def simcse_sup_loss(y_pred: 'tensor',t_pred:'tensor') -> 'tensor':
    """有监督的损失函数
    y_pred (tensor): bert的输出, [batch_size * 3, 768]
    
    """
    # 得到y_pred对应的label, 每第三句没有label, 跳过, label= [1, 0, 4, 3, ...]
    # y_true = torch.arange(y_pred.shape[0], device=DEVICE)
    # print(y_true)
    # use_row = torch.where((y_true + 1) % 3 != 0)[0]
    # y_true = (use_row - use_row % 3 * 2) + 1
    y_pred=torch.cat((y_pred,t_pred),dim=0)
    y_true=[]
    for i in range(y_pred.size()[0]):
        if i<y_pred.size()[0]//2:
            y_true.append(i+y_pred.size()[0]//2)
        else:
            y_true.append(i-y_pred.size()[0]//2)
    # print(y_true)
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    # print(y_pred.size())
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # print(sim.size())
    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(sim.size()[0], device=DEVICE) * 1e12
    # 选取有效的行
    # sim = torch.index_select(sim, 0, use_row)
    # print(sim.size())
    # 相似度矩阵除以温度系数
    sim = sim / 0.05
    # 计算相似度矩阵与y_true的交叉熵损失
    # print(torch.tensor(y_true).to(DEVICE))
    loss = F.cross_entropy(sim, torch.tensor(y_true).to(DEVICE))
    return loss
    
def word_att_loss(whole_word_att,contact_att):
    whole_word_att_2_begin = whole_word_att[:,:,0,:,:]
    whole_word_att_3_begin = whole_word_att[:,:,1,:,:]
    half_head_att_down = contact_att[6:,:,:6,:,:]
    whole_word_att_n = half_head_att_down.sum(dim=-3)
    # print(whole_word_att_n)
    whole_word_att_2n_b = whole_word_att_n.masked_select(whole_word_att_2_begin)
    whole_word_att_2n_e = whole_word_att_n.masked_select(torch.nn.functional.pad(whole_word_att_2_begin,pad=(1,0),mode="constant")[:,:,:,:-1])
    whole_word_att_3n_b = whole_word_att_n.masked_select(whole_word_att_3_begin)
    whole_word_att_3n_m = whole_word_att_n.masked_select(torch.nn.functional.pad(whole_word_att_3_begin,pad=(1,0),mode="constant")[:,:,:,:-1])
    whole_word_att_3n_e = whole_word_att_n.masked_select(torch.nn.functional.pad(whole_word_att_3_begin,pad=(2,0),mode="constant")[:,:,:,:-2])
    loss_1 = torch.nn.SmoothL1Loss(reduction="mean")(whole_word_att_2n_b,whole_word_att_2n_e)
    loss_2 = torch.nn.SmoothL1Loss(reduction="mean")(whole_word_att_3n_b,whole_word_att_3n_m)
    loss_3 = torch.nn.SmoothL1Loss(reduction="mean")(whole_word_att_3n_m,whole_word_att_3n_e)
    return loss_1+loss_2+loss_3


def eval(model, dataloader,threshold) -> float:
    """模型评估函数 
    批量预测, 计算cos_sim, 转成numpy数组拼接起来, 一次性求spearman相关度
    """
    model.eval()
    sim_tensor = torch.tensor([], device=DEVICE)
    label_array = np.array([])
    with torch.no_grad():
        for source, target, label in dataloader:
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source['input_ids'].squeeze(1).to(DEVICE)
            # print(source_input_ids)
            source_attention_mask = source['attention_mask'].squeeze(1).to(DEVICE)
            source_token_type_ids = source['token_type_ids'].squeeze(1).to(DEVICE)
            # print(source_token_type_ids)
            source_pred,_ = model(source_input_ids, source_attention_mask, source_token_type_ids,None)
            # target        [batch, 1, seq_len] -> [batch, seq_len]
            target_input_ids = target['input_ids'].squeeze(1).to(DEVICE)
            target_attention_mask = target['attention_mask'].squeeze(1).to(DEVICE)
            target_token_type_ids = target['token_type_ids'].squeeze(1).to(DEVICE)
            target_pred,_ = model(target_input_ids, target_attention_mask, target_token_type_ids,None)
            # concat

            sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)
            label_array = np.append(label_array, np.array(label))
        spear_cor=spearmanr(label_array, sim_tensor.cpu().numpy()).correlation
        out_put_sim=sim_tensor.clone()
        sim_tensor[sim_tensor>=threshold]=1
        sim_tensor[sim_tensor<threshold]=0
        label_array[label_array>=1]=1
        label_array[label_array<1]=0
        acc=metrics.accuracy_score(label_array, sim_tensor.cpu().numpy())
    # corrcoef
    return spear_cor,acc


'''
CLS_Selcet mix_cls
2022-04-02 12:32:25.733 | INFO     | __main__:<module>:391 - lqcmc:simcse+word_att:cls:dev_corrcoef: 0.4851 dev_acc: 0.0027
2022-04-02 12:32:25.733 | INFO     | __main__:<module>:392 - lqcmc:simcse+word_att:cls:test_corrcoef: 0.5753 test_acc: 0.0003
2022-04-02 12:48:17.471 | INFO     | __main__:<module>:391 - lqcmc:simcse:cls:dev_corrcoef: 0.4644dev_acc: 0.0022
2022-04-02 12:48:17.472 | INFO     | __main__:<module>:392 - lqcmc:simcse:cls:test_corrcoef: 0.5489test_acc: 0.0003
2022-04-02 12:16:02.228 | INFO     | __main__:<module>:391 - pawsx:simcse:cls:dev_corrcoef: 0.0291dev_acc: 0.0000
2022-04-02 12:16:02.228 | INFO     | __main__:<module>:392 - pawsx:simcse:cls:test_corrcoef: 0.1003test_acc: 0.0000
2022-04-02 12:10:41.474 | INFO     | __main__:<module>:391 - pawsx:simcse+word_att:cls:dev_corrcoef: 0.0311dev_acc: 0.0000
2022-04-02 12:10:41.474 | INFO     | __main__:<module>:392 - pawsx:simcse+word_att:cls:test_corrcoef: 0.1019test_acc: 0.0000
2022-04-02 12:05:12.007 | INFO     | __main__:<module>:391 - sts:simcse:cls:dev_corrcoef: 0.7142dev_acc: 0.0336
2022-04-02 12:05:12.007 | INFO     | __main__:<module>:392 - sts:simcse:cls:test_corrcoef: 0.6547test_acc: 0.0235
2022-04-02 12:01:37.879 | INFO     | __main__:<module>:391 - sts:simcse+word_att:cls:dev_corrcoef: 0.7167dev_acc: 0.1008
2022-04-02 12:01:37.880 | INFO     | __main__:<module>:392 - sts:simcse+word_att:cls:test_corrcoef: 0.6595test_acc: 0.0551
2022-04-02 14:03:36.971 | INFO     | __main__:<module>:391 - atec:simcse+word_att:cls:dev_corrcoef: 0.3963dev_acc: 0.1183
2022-04-02 14:03:36.971 | INFO     | __main__:<module>:392 - atec:simcse+word_att:cls:test_corrcoef: 0.3850test_acc: 0.1206
'''

'''
bert cls

2022-04-02 14:48:17.693 | INFO     | __main__:<module>:391 - sts:simcse+word_att:cls:dev_corrcoef: 0.7707dev_acc: 0.2016
2022-04-02 14:48:17.693 | INFO     | __main__:<module>:392 - sts:simcse+word_att:cls:test_corrcoef: 0.6991test_acc: 0.1139

2022-04-02 14:50:35.873 | INFO     | __main__:<module>:391 - sts:simcse:cls:dev_corrcoef: 0.7693dev_acc: 0.2154
2022-04-02 14:50:35.873 | INFO     | __main__:<module>:392 - sts:simcse:cls:test_corrcoef: 0.7002test_acc: 0.1330

2022-04-02 14:54:03.115 | INFO     | __main__:<module>:391 - pawsx:simcse+word_att:cls:dev_corrcoef: 0.0492dev_acc: 0.0135
2022-04-02 14:54:03.115 | INFO     | __main__:<module>:392 - pawsx:simcse+word_att:cls:test_corrcoef: 0.1140test_acc: 0.0130

2022-04-02 14:56:04.451 | INFO     | __main__:<module>:391 - pawsx:simcse:cls:dev_corrcoef: 0.0487dev_acc: 0.0125
2022-04-02 14:56:04.451 | INFO     | __main__:<module>:392 - pawsx:simcse:cls:test_corrcoef: 0.1158test_acc: 0.0135


'''
def cosent_loss(y_true, y_pred,y_target):
    # print(y_true)
    # 2. 对输出的句子向量进行l2归一化   后面只需要对应为相乘  就可以得到cos值了
    norms = (y_pred ** 2).sum(axis=1, keepdims=True) ** 0.5
    # y_pred = y_pred / torch.clip(norms, 1e-8, torch.inf)
    y_pred = y_pred / norms

    norms = (y_target ** 2).sum(axis=1, keepdims=True) ** 0.5
    # y_pred = y_pred / torch.clip(norms, 1e-8, torch.inf)
    y_target = y_target / norms
    # print(y_target.size())
    # 3. 奇偶向量相乘
    # print((y_pred * y_target).size())
    # print(F.cosine_similarity(y_pred, y_target, dim=-1).size())
    y_pred = torch.sum(y_pred * y_target, dim=1) * 20
    # y_pred = F.cosine_similarity(y_pred, y_target, dim=-1) * 20
    # print("cosent_sim:",y_pred)
    y_pred_true = y_pred[y_true.bool()]
    y_pred_true = 0.85-y_pred_true
    y_pred_true = y_pred_true.view(-1)
    # print(y_pred_true.size())
    # 4. 取出负例-正例的差值
    y_pred = y_pred[:, None] - y_pred[None, :]  # 这里是算出所有位置 两两之间余弦的差值
    # 矩阵中的第i行j列  表示的是第i个余弦值-第j个余弦值
    y_true = y_true[:, None] < y_true[None, :]   # 取出负例-正例的差值
    y_true = y_true.float()
    y_pred = y_pred - (1 - y_true) * 1e12
    # print("cosent:",torch.masked_select(y_pred,y_true.bool()))
    y_pred = y_pred.view(-1)

    if torch.cuda.is_available():
        y_pred = torch.cat((torch.tensor([0]).float().cuda(), y_pred), dim=0)  # 这里加0是因为e^0 = 1相当于在log中加了1
        y_pred_true_loss = torch.cat((torch.tensor([0]).float().cuda(),y_pred_true ), dim=0)
    else:
        y_pred = torch.cat((torch.tensor([0]).float(), y_pred), dim=0)  # 这里加0是因为e^0 = 1相当于在log中加了1
        y_pred_true_loss = torch.cat((torch.tensor([0]).float(),y_pred_true ), dim=0)

    y_pred_true_loss=torch.logsumexp(y_pred_true_loss, dim=0)
    return torch.logsumexp(y_pred, dim=0)+y_pred_true_loss

def train(model, antisimtrain_dl,simtrain_dl, optimizer,loss_type) -> None:
    """模型训练函数 
    """
    model.train()
    global best
    all_loss=0
    sim_bath_iterator=iter(simtrain_dl)
    for batch_idx, (assource,astarget,aslabel,asclasslabel) in enumerate(tqdm(antisimtrain_dl), start=1):
        # source        [batch, 1, seq_len] -> [batch, seq_len]
        # print(source_token_type_ids)
        try:
            ssource,starget,slabel,sclasslabel=next(sim_bath_iterator)
        except StopIteration:
            sim_bath_iterator=iter(simtrain_dl)
            ssource,starget,slabel,sclasslabel=next(sim_bath_iterator)
        # print(sclasslabel.size())
        ssource_input_ids = ssource['input_ids'].squeeze(1).to(DEVICE)
        ssource_attention_mask = ssource['attention_mask'].squeeze(1).to(DEVICE)
        ssource_token_type_ids = ssource['token_type_ids'].squeeze(1).to(DEVICE)
        starget_input_ids = starget['input_ids'].squeeze(1).to(DEVICE)
        starget_attention_mask = starget['attention_mask'].squeeze(1).to(DEVICE)
        starget_token_type_ids = starget['token_type_ids'].squeeze(1).to(DEVICE)

        assource_input_ids = assource['input_ids'].squeeze(1).to(DEVICE)
        assource_attention_mask = assource['attention_mask'].squeeze(1).to(DEVICE)
        assource_token_type_ids = assource['token_type_ids'].squeeze(1).to(DEVICE)
        astarget_input_ids = astarget['input_ids'].squeeze(1).to(DEVICE)
        astarget_attention_mask = astarget['attention_mask'].squeeze(1).to(DEVICE)
        astarget_token_type_ids = astarget['token_type_ids'].squeeze(1).to(DEVICE)


        source_input_ids=torch.cat((assource_input_ids,astarget_input_ids),dim=0)
        source_attention_mask=torch.cat((assource_attention_mask,astarget_attention_mask),dim=0)
        source_token_type_ids=torch.cat((assource_token_type_ids,astarget_token_type_ids),dim=0)
        # target        [batch, 1, seq_len] -> [batch, seq_len]

        target_input_ids=torch.cat((ssource_input_ids,starget_input_ids),dim=0)
        target_attention_mask=torch.cat((ssource_attention_mask,starget_attention_mask),dim=0)
        target_token_type_ids=torch.cat((ssource_token_type_ids,starget_token_type_ids),dim=0)
        ssclass_label=sclasslabel.repeat(1,2).view(-1).to(DEVICE)
        asclass_label= torch.cat((asclasslabel[::2],asclasslabel[1::2]),dim=-1).to(DEVICE)

        ss_pred,ssclass_linear= model(target_input_ids, target_attention_mask, target_token_type_ids,ssclass_label)

        ass_pred,asclass_linear= model(source_input_ids, source_attention_mask, source_token_type_ids,asclass_label)

        # print(ss_pred.size(),ass_pred.size())
        pred_source=torch.cat((ss_pred[:ss_pred.size()[0]//2],ass_pred[:ass_pred.size()[0]//2]),dim=0)
        pred_target=torch.cat((ss_pred[ss_pred.size()[0]//2:],ass_pred[ass_pred.size()[0]//2:]),dim=0)
        aslabel=aslabel.to(DEVICE)
        slabel=slabel.to(DEVICE)

        # print(class_label.size())
        label=torch.cat((slabel,aslabel),dim=0)
        # print(class_linear.size())
        class_loss = torch.nn.CrossEntropyLoss()
        loss1=class_loss(ssclass_linear,ssclass_label)
        loss2=class_loss(asclass_linear,asclass_label)
        # print(loss1)
        loss = cosent_loss(label,pred_source,pred_target)+loss1+loss2

        all_loss+=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 评估
    return all_loss
    
def train_logic():
    SNIL_TRAIN = {"nlu":train_set,"sts":'./datasets/cnsd-sts-train.csv',"lqcmc":"./datasets/LCQMC.train.data.csv","atec":"./datasets/ATEC.train.data.csv","bq":"./datasets/BQ.train.data.csv","pawsx":"./datasets/pawsx.train.csv"}
    STS_DEV = {"nlu":"./dialogue_test.csv","sts":'./datasets/cnsd-sts-dev.csv',"lqcmc":"./datasets/LCQMC.valid.data.csv","atec":"./datasets/ATEC.valid.data.csv","bq":"./datasets/BQ.valid.data.csv","pawsx":"./datasets/pawsx.dev.csv"}
    STS_TEST = {"afqmc":"./datasets/afqmc.dev.csv","sts":'./datasets/cnsd-sts-test.csv',"lqcmc":"./datasets/LCQMC.test.data.csv","atec":"./datasets/ATEC.test.data.csv","bq":"./datasets/BQ.valid.data.csv","pawsx":"./datasets/pawsx.test.csv"}
    for data_name in ["nlu"]:
        # load data
        train_data = load_data(data_name, SNIL_TRAIN[data_name],"dev")
        # random.shuffle(train_data)
        dev_data = load_data(data_name, SNIL_TRAIN[data_name],"dev")
        test_data = load_data(data_name, STS_DEV[data_name],"test")
        label_num = load_data(data_name,SNIL_TRAIN[data_name],"label_num")
        # train_dataloader = DataLoader(TrainDataset(train_data), batch_size=BATCH_SIZE)
        data_key = list(set([raw[0] for raw in train_data if isinstance(raw[0],str)]))
        astrain_dataloader = DataLoader(AntiSimTrainDataset(train_data,data_key), batch_size=BATCH_SIZE,collate_fn=collate_fn,shuffle=True)
        strain_dataloader = DataLoader(SimTrainDataset(dev_data,data_key), batch_size=BATCH_SIZE,collate_fn=collate_fn,shuffle=True)
        test_dataloader = DataLoader(TestDataset(test_data), batch_size=BATCH_SIZE,collate_fn=collate_fn)
        threshold=0.8
        # load model
        for loss_type in ["simcse","simcse+word_att"]:
            for POOLING in ['cls','mix_cls','pooler']:#, 'last-avg', 'first-last-avg']:
                logger.info(f'device: {DEVICE}, pooling: {POOLING}, model path: {model_path}')
                model = SimcseModel(pretrained_model=model_path, pooling=POOLING,label_num=label_num)
                model.to(DEVICE)
                optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
                # train
                best = 1e10
                early_stop_batch = 0
                early_stop=False
                for epoch in range(EPOCHS):
                    logger.info(f'epoch: {epoch}')
                    loss=train(model, astrain_dataloader, strain_dataloader, optimizer,loss_type)
                    corrcoef,acc = eval(model, test_dataloader,threshold)
                    logger.info(f'loss: {loss:.4f} acc:{acc:.4f} corrcoef:{corrcoef:.4f} ')
                    if round(loss,0) < best:
                        early_stop_batch = 0
                        best = round(loss,0)
                        torch.save(model.state_dict(), SAVE_PATH%(epoch,loss_type,POOLING))
                    else:
                        early_stop_batch+=1
                    if early_stop_batch>=EARLY:
                        early_stop=True
                        logger.info(f'train is early stop, best model is saved at {SAVE_PATH%(EPOCHS-1,loss_type,POOLING)}')
                        # eval
                        model.load_state_dict(torch.load(SAVE_PATH%(EPOCHS-1,loss_type,POOLING)))
                        test_corrcoef,test_acc = eval(model, test_dataloader,threshold)
                        logger.info(data_name+":"+loss_type+":"+POOLING+":"+f'test_corrcoef: {test_corrcoef:.4f}'+f'test_acc: {test_acc:.4f}')
                        break
                if not early_stop:
                    logger.info(f'train is finished, best model is saved at {SAVE_PATH%(EPOCHS-1,loss_type,POOLING)}')
                    # eval
                    model.load_state_dict(torch.load(SAVE_PATH%(EPOCHS-1,loss_type,POOLING)))
                    test_corrcoef,test_acc = eval(model, test_dataloader,threshold)
                    logger.info(data_name+":"+loss_type+":"+POOLING+":"+f'test_corrcoef: {test_corrcoef:.4f}'+f'test_acc: {test_acc:.4f}')
class threshold_struct(nn.Module):
    def __init__(self,input_size):
        super(threshold_struct, self).__init__()
        self.linear = nn.Linear(6,1)
        self.data_linear = nn.Linear(input_size,3)
        self.tanh = nn.Tanh()
    def smooth_tanh(self,x):
        return (2/(1+torch.exp(-x)))-1
    def forward(self, input_ids, target_ids):
        # threshold=self.linear(torch.cat((input_ids,target_ids,input_ids-target_ids),dim=-1))
        # print(torch.cat((F.cosine_similarity(input_ids,target_ids,dim=-1).unsqueeze(dim=-1),(1-F.cosine_similarity(input_ids,target_ids,dim=-1)).unsqueeze(dim=-1)),dim=-1).size())
        data_feat=self.data_linear(torch.cat((input_ids,target_ids,input_ids-target_ids),dim=-1))
        data_feat=self.tanh(data_feat)
        threshold=self.linear(torch.cat(((torch.nn.functional.normalize(input_ids, dim=-1, p=2)*torch.nn.functional.normalize(target_ids, dim=-1, p=2)).sum(dim=-1).unsqueeze(dim=-1),
                                         (1-F.cosine_similarity(input_ids,target_ids,dim=-1)).unsqueeze(dim=-1),
                                         F.cosine_similarity(input_ids,target_ids,dim=-1).unsqueeze(dim=-1),
                                         data_feat
                                         ),dim=-1))
        threshold=self.smooth_tanh(threshold)
        return threshold

def dev_threshold(model,threshold_model,test_dataloader,epoch):
    model.eval()
    sim_tensor = torch.tensor([], device=DEVICE)
    label_array = np.array([])
    all_threshold = torch.tensor([], device=DEVICE)
    with torch.no_grad():
        for source, target, label,whole_att_source,whole_att_target in tqdm(test_dataloader):
            source_input_ids = source['input_ids'].squeeze(1).to(DEVICE)
            # print(source_input_ids)
            source_attention_mask = source['attention_mask'].squeeze(1).to(DEVICE)
            source_token_type_ids = source['token_type_ids'].squeeze(1).to(DEVICE)
            # print(source_token_type_ids)
            source_pred= model(source_input_ids, source_attention_mask, source_token_type_ids)
            # target        [batch, 1, seq_len] -> [batch, seq_len]
            target_input_ids = target['input_ids'].squeeze(1).to(DEVICE)
            target_attention_mask = target['attention_mask'].squeeze(1).to(DEVICE)
            target_token_type_ids = target['token_type_ids'].squeeze(1).to(DEVICE)
            target_pred= model(target_input_ids, target_attention_mask, target_token_type_ids)
            source_pred=source_pred.detach()
            target_pred=target_pred.detach()
            threshold=threshold_model(source_pred,target_pred).view(-1)
            sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)
            all_threshold = torch.cat((all_threshold,threshold),dim=0)
            label_array = np.append(label_array, np.array(label))
        sim=sim_tensor.clone().cpu().numpy()
        sim_tensor[sim_tensor>=all_threshold]=1
        sim_tensor[sim_tensor<all_threshold]=0
        label_array[label_array>=1]=1
        label_array[label_array<1]=0
        acc=metrics.accuracy_score(label_array, sim_tensor.cpu().numpy())
        sim_array=np.array([[i,raw] for i,raw in enumerate(sim)])
        sim_array=np.vstack((sim_array,np.array([[i,raw] for i,raw in enumerate(all_threshold.clone().cpu().numpy())])))
        label=np.hstack((label_array + 1, np.array([3] * len(sim)))).astype(int)
    return acc

def train_threshold(model,threshold_model,test_dataloader,astrain_dataloader,optimizer):
    threshold_model.train()
    all_loss=0
    anti_bath_iterator=iter(astrain_dataloader)
    for source, target, label,whole_att_source,whole_att_target in tqdm(test_dataloader):
        label = label.to(DEVICE)
        source_input_ids = source['input_ids'].squeeze(1).to(DEVICE)
        # print(source_input_ids)
        source_attention_mask = source['attention_mask'].squeeze(1).to(DEVICE)
        source_token_type_ids = source['token_type_ids'].squeeze(1).to(DEVICE)
        # print(source_token_type_ids)
        source_pred,_ = model(source_input_ids, source_attention_mask, source_token_type_ids)
        # target        [batch, 1, seq_len] -> [batch, seq_len]
        target_input_ids = target['input_ids'].squeeze(1).to(DEVICE)
        target_attention_mask = target['attention_mask'].squeeze(1).to(DEVICE)
        target_token_type_ids = target['token_type_ids'].squeeze(1).to(DEVICE)
        target_pred,_ = model(target_input_ids, target_attention_mask, target_token_type_ids)
        try:
            asource,atarget,alabel,awhole_att_source,awhole_att_target=next(anti_bath_iterator)
        except StopIteration:
            anti_bath_iterator=iter(astrain_dataloader)
            asource,atarget,alabel,awhole_att_source,awhole_att_target=next(anti_bath_iterator)
        alabel=alabel.to(DEVICE)
        asource_input_ids = asource['input_ids'].squeeze(1).to(DEVICE)
        # print(source_input_ids)
        asource_attention_mask = asource['attention_mask'].squeeze(1).to(DEVICE)
        asource_token_type_ids = asource['token_type_ids'].squeeze(1).to(DEVICE)
        # print(source_token_type_ids)
        asource_pred,_ = model(asource_input_ids, asource_attention_mask, asource_token_type_ids)
        atarget_input_ids = atarget['input_ids'].squeeze(1).to(DEVICE)
        atarget_attention_mask = atarget['attention_mask'].squeeze(1).to(DEVICE)
        atarget_token_type_ids = atarget['token_type_ids'].squeeze(1).to(DEVICE)
        atarget_pred,_ = model(atarget_input_ids, atarget_attention_mask, atarget_token_type_ids)

        source_pred=torch.cat((source_pred.detach(),asource_pred.detach()),dim=0)
        target_pred=torch.cat((target_pred.detach(),atarget_pred.detach()),dim=0)
        label=torch.cat((label,alabel),dim=0)
        threshold=threshold_model(source_pred,target_pred).view(-1)
        sim_thres=threshold[:BATCH_SIZE].sum()
        unsim_thres=threshold[BATCH_SIZE:].sum()
        thres_std = threshold[:BATCH_SIZE].std() + threshold[BATCH_SIZE:].std()
        sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
        dist = torch.abs(label-threshold)-torch.abs(label-sim)
        thres_margin=torch.exp(-torch.abs(sim_thres-unsim_thres))
        dist = dist*20
        dist = torch.cat((torch.tensor([0]).float().to(DEVICE), dist), dim=0)  # 这里加0是因为e^0 = 1相当于在log中加了1
        loss=torch.logsumexp(dist, dim=0)+thres_margin+thres_std
        all_loss+=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return all_loss
def auto_threshold(train_dataloader,astrain_dataloader,model,dev_dataloader):
    threshold_model= threshold_struct(768*3).to(DEVICE)
    optimizer = torch.optim.AdamW(threshold_model.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        logger.info(f'epoch: {epoch}')
        all_loss=train_threshold(model,threshold_model,train_dataloader,astrain_dataloader,optimizer)
        acc = dev_threshold(model,threshold_model,dev_dataloader,epoch)
        logger.info(f'loss: {all_loss:.4f} acc:{acc:.4f}')
        torch.save(threshold_model.state_dict(), SAVE_PATH%(epoch,"threshold","linear"))
def threshold_predict(model,dev_dataloader,epoch):
    threshold_model= threshold_struct(768*3).to(DEVICE)
    threshold_model.to(DEVICE)
    threshold_model.load_state_dict(torch.load(SAVE_PATH%(epoch,"threshold","linear")))
    acc= dev_threshold(model,threshold_model,dev_dataloader,epoch)
    logger.info(f'auto threshold acc:{acc:.4f}')


if __name__ == '__main__':
    train_logic()
# if __name__ == '__main__':
#     STS_DEV = {"nlu":"./datasets/eval_data.csv"}
#     STS_TRAIN = {"nlu":"./datasets/train_data.csv"}
#     viz=Visdom(env='auto_threshold_margrin_consin_normdot_datafeat_20T',server="http://10.231.135.108",port=8097)
#     for data_name in ["nlu"]:
#         for POOLING in ['cls','mix_cls','pooler']:
#             threshold_data = load_data(data_name, STS_DEV[data_name],"threshold")
#             threshold_train_ori= load_data(data_name, STS_TRAIN[data_name],"threshold")
#             # threshold_train,threshold_dev=train_test_split(threshold_data, test_size = 0.5,random_state=1)
#             train_dataloader = DataLoader(TestDataset(threshold_train_ori), batch_size=BATCH_SIZE)
#             astrain_dataloader = DataLoader(AntiSimTrainDataset(threshold_train_ori), batch_size=BATCH_SIZE)
#             # dev_dataloader = DataLoader(TestDataset(threshold_dev), batch_size=BATCH_SIZE)
#             dev_dataloader = DataLoader(TestDataset(threshold_data), batch_size=BATCH_SIZE)
#             test_model_path="./saved_model/nlu_cosent_like_uda_47_simcse_cls_2.pt"
#             model = SimcseModel(pretrained_model=model_path, pooling=POOLING)
#             model.to(DEVICE)
#             model.load_state_dict(torch.load(test_model_path))
#             force_threshold_step=10
#             threshold_list = [i/force_threshold_step for i in range(force_threshold_step)]
#             color = np.random.randint(0, 255, (3, 3,))
#             # for threshold in threshold_list:
#             #     test_corrcoef,test_acc,sim,label = eval(model, dev_dataloader,threshold)
#             #     sim_array=np.array([[i,raw] for i,raw in enumerate(sim)])
#             #     sim_array=np.vstack((sim_array,np.array([[i,threshold] for i in range(len(sim))])))
#             #     # print(sim_array)
#             #     label=np.hstack((label + 1, np.array([3] * len(sim)))).astype(int)
#             #     # print(label,label.dtype)
#             #     viz.scatter(X=sim_array,Y=label,win="auto_threshold",opts=dict(title="force_threshold",markersize=5,markercolor=color))
#                 # print("threshold:%s acc:%s"%(threshold,test_acc))
#             auto_threshold(train_dataloader,astrain_dataloader,model,dev_dataloader)
            # threshold_predict(model,dev_dataloader,38)
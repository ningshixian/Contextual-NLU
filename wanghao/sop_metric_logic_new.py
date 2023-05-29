#!/usr/bin/python3  
# -*- coding: utf-8 -*-
# @Time    : 2022/8/17 9:58
# @Author  : wanghao27
# @Site    : 
# @File    : sop_metric_data.py
# @Email   : wanghao27@longfor.com
import re

import torch

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from trainer_sup import collate_fn, SimcseModel, pad_to_maxlen
from utils.oss_cfg import OSS2
from typing import List


BATCH_SIZE = 64
LR = 2e-5
MAXLEN = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path="./saved_model/seq128_b64_class_rand_true_notest_amsoft_trueloss_nlu_cosent_like_uda_1_simcse_cls_2.pt"
# pretrain_model="/data/wanghao/work/pretrained_models/simbert-chinese-base"
pretrain_model="/data/wanghao/work/pretrained_models/roberta_chinese_wwm_ext"
tokenizer = BertTokenizer.from_pretrained(pretrain_model)

def collate_fn(batch):
    # 按batch进行padding获取当前batch中最大长度
    max_len=0
    for raw in batch:
        seq_len=len(raw["input_ids"])
        if seq_len>max_len:
            max_len=seq_len

    if max_len > MAXLEN:
        max_len = MAXLEN

    # 定一个全局的max_len
    # max_len = 128

    sentence_1={}
    for item in batch:
        if "input_ids" not in sentence_1:
            sentence_1["input_ids"]=[pad_to_maxlen(item['input_ids'], max_len=max_len)]
        else:
            sentence_1["input_ids"].append(pad_to_maxlen(item['input_ids'], max_len=max_len))
        if "attention_mask" not in sentence_1:
            sentence_1["attention_mask"]=[pad_to_maxlen(item['attention_mask'], max_len=max_len)]
        else:
            sentence_1["attention_mask"].append(pad_to_maxlen(item['attention_mask'], max_len=max_len))
        if "token_type_ids" not in sentence_1:
            sentence_1["token_type_ids"]=[pad_to_maxlen(item['token_type_ids'], max_len=max_len)]
        else:
            sentence_1["token_type_ids"].append(pad_to_maxlen(item['token_type_ids'], max_len=max_len))
    for raw in sentence_1:
        sentence_1[raw]=torch.tensor(sentence_1[raw],dtype=torch.long)

    return sentence_1

class PriDataset(Dataset):
    """训练数据集, 重写__getitem__和__len__方法
    """
    def __init__(self, data: List):
        self.data = [idata for idata in data if isinstance(idata,str) ]
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
        return self.text_2_id(line)

def make_primary_vec(path_or_list,model):
    if isinstance(path_or_list,str):
        primary_data=list(pd.read_excel(path_or_list)["primary_question"])
    else:
        primary_data=path_or_list
    know_data=pd.read_excel("knowledge_data_pro.xlsx")
    primary_q=list([raw.replace(" ","") for raw in know_data["primary_question"]])
    similar_q=list(know_data["similar_question"])

    out_primary_data=[]
    primary_vec=torch.tensor([]).to(DEVICE)
    similar_data=[]
    for raw in primary_data:
        if raw in primary_q:
            primary_index=primary_q.index(raw)
            similar_list=similar_q[primary_index].split("###")
            out_primary_data.extend([raw]*(len(similar_list)+1))
            similar_data.extend([raw]+similar_list)
        else:
            similar_list=[]
            out_primary_data.extend([raw])
        dataloader = DataLoader(PriDataset([raw]+similar_list), batch_size=BATCH_SIZE,collate_fn=collate_fn)
        primary_vec=torch.cat((primary_vec,model_pridect(model,dataloader)),dim=0)
    assert primary_vec.size()[0]==len(out_primary_data)
    return out_primary_data,primary_vec,similar_data


def model_pridect(model, dataloader):
    model.eval()
    all_data_vector=torch.tensor([]).to(DEVICE)
    with torch.no_grad():
        for source in tqdm(dataloader):
            source_input_ids = source['input_ids'].squeeze(1).to(DEVICE)
            source_attention_mask = source['attention_mask'].squeeze(1).to(DEVICE)
            source_token_type_ids = source['token_type_ids'].squeeze(1).to(DEVICE)
            source_pred,_ = model(source_input_ids, source_attention_mask, source_token_type_ids)
            all_data_vector=torch.cat((all_data_vector,source_pred),dim=0)

    return all_data_vector

def dialogue_pridect(path,model,primary_data,primary_vec,thres,similar_data,predict_save_path,sop_text_name,topN=5):
    sop_test_data=pd.read_excel(path)
    context_id=sop_test_data["context_id"]
    context_data=sop_test_data["context_data"]
    context_label=sop_test_data["context_label"]
    context_pridect=[]
    context_pridect_score=[]
    context_sop={}
    context_top_sop={}
    curr_context_id=""
    context_key_sop={}
    with open(predict_save_path,"w",encoding="utf-8") as f:
        for i,raw in enumerate(context_data):
            # print(context_id[i])
            if isinstance(context_id[i],str):
                curr_context_id=context_id[i]
                context_sop[curr_context_id]=[]
                context_top_sop[curr_context_id]=[]
                if "_begin" not in raw and "None" not in raw:
                    idata=re.sub("[http|https]://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+","",raw)
                    idata=re.sub("(<img).*(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+","<img",idata)
                    dialogue_input_data=[idata]
                else:
                    dialogue_input_data=[]
            else:
                if "_begin" not in raw and "None" not in raw:
                    idata=re.sub("[http|https]://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+","",raw)
                    idata=re.sub("(<img).*(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+","<img",idata)
                    dialogue_input_data.extend([idata])
                else:
                    context_pridect.append(None)
                    context_pridect_score.append(None)
                    continue
            if len(dialogue_input_data)>0:
                dataloader = DataLoader(PriDataset(["/".join(dialogue_input_data[-5:])]), batch_size=BATCH_SIZE,collate_fn=collate_fn)
                one_vector=model_pridect(model,dataloader)
                sop_cos=F.cosine_similarity(one_vector,primary_vec)
                sop_cos_argmax=sop_cos.argmax()
                sop_cos_max=sop_cos.max()
                if sop_cos_max>thres:
                    if curr_context_id in sop_text_name:
                        if dialogue_input_data[-1] in list(sop_text_name[curr_context_id].keys()):
                            if curr_context_id not in context_key_sop:
                                context_key_sop[curr_context_id]={dialogue_input_data[-1]:primary_data[sop_cos_argmax]}
                            else:
                                context_key_sop[curr_context_id].update({dialogue_input_data[-1]:primary_data[sop_cos_argmax]})
                    context_pridect.append(primary_data[sop_cos_argmax])
                    context_pridect_score.append(sop_cos_max.cpu())
                    context_sop[curr_context_id].append(primary_data[sop_cos_argmax])
                    if len(dialogue_input_data)<topN:
                        context_top_sop[curr_context_id].append(primary_data[sop_cos_argmax])
                    print(primary_data[sop_cos_argmax],":",sop_cos_max.cpu(),":","/".join(dialogue_input_data[-5:]),":",similar_data[sop_cos_argmax])
                    f.write(primary_data[sop_cos_argmax]+":"+str(sop_cos_max.cpu().item())+":"+"/".join(dialogue_input_data[-5:])+":"+similar_data[sop_cos_argmax]+"\n")
                else:
                    # if dialogue_input_data[-1] in list(sop_text_name[curr_context_id].keys()):
                    #     if curr_context_id not in context_key_sop:
                    #         context_key_sop[curr_context_id]={dialogue_input_data[-1]:None}
                    #     else:
                    #         context_key_sop[curr_context_id].update({dialogue_input_data[-1]:None})
                    context_pridect.append(None)
                    context_pridect_score.append(None)
            else:
                context_pridect.append(None)
                context_pridect_score.append(None)
    assert len(context_pridect)==len(context_id),"context_pridect_len:%s,context_id_len:%s"%(len(context_pridect),len(context_id))

    pridect_data=pd.DataFrame({"context_id":context_id,"context_data":context_data,"context_label":context_label,"context_predict":context_pridect,"predict_score":context_pridect_score})

    pridect_data.to_excel("sop_context_nlu_predict.xlsx")

    return context_sop,context_top_sop,context_key_sop
def context_nlu_metric(sop_predict,sop_true):
    all_true=len(list(sop_true.keys()))
    true_num=0
    bad_context_id={}
    for raw in sop_true:
        sop_tru=list(set(sop_true[raw]))
        if len(sop_tru)>1:
            sop_tru=[raw for raw in sop_tru if raw!=None]
        if raw in sop_predict:
            sop_pre=list(set(sop_predict[raw])) if sop_predict[raw]!=[] else [None]
        else:
            sop_pre=[None]
        if sop_tru==sop_pre:
            true_num+=1
        else:
            bad_context_id[raw]={"true":sop_tru,"pred":sop_pre}
    print("context_nlu_acc:%s"%str(true_num/all_true))
    return bad_context_id

def nlu_metric(nlu_predict,sop_true):
    all_true=len(list(sop_true.keys()))
    true_num=0
    bad_context_id={}
    for raw in sop_true:
        sop_tru=list(set(sop_true[raw]))
        if len(sop_tru)>1:
            sop_tru=[raw for raw in sop_tru if raw!=None]
        if raw in nlu_predict:
            sop_pre=list(set(nlu_predict[raw])) if nlu_predict[raw]!=[] else [None]
        else:
            sop_pre=[None]
        if sop_tru==sop_pre:
            true_num+=1
        else:
            bad_context_id[raw]={"true":sop_tru,"pred":sop_pre}
    print("nlu_acc:%s"%str(true_num/all_true))
    return bad_context_id
def read_true_sop(true_path):
    true_sop=pd.read_excel(true_path)
    print(true_sop.keys())
    context_id=true_sop["会话ID（context_id）"]
    context_sop_name=true_sop["触发SOP名称"]
    context_sop_text=true_sop["触发文本"]
    context_label=true_sop["重新标注"]
    nlu_predict=true_sop["nlu预测"]
    sop_true={}
    sop_nlu_predict={}
    sop_text_name={}
    nlu_predict_true=[(context_id[i],context_sop_name[i]) for i,raw in enumerate(nlu_predict) if raw=="是"]
    nlu_predict_false=[(context_id[i],None) for i,raw in enumerate(nlu_predict) if raw=="否"]
    s_true=[(context_id[i],context_sop_name[i],context_sop_text[i]) for i,raw in enumerate(context_label) if raw=="是"]
    s_false=[(context_id[i],None,context_sop_text[i]) for i,raw in enumerate(context_label) if raw=="否"]
    for raw in s_true+s_false:
        if raw[0] not in sop_true:
            sop_true[raw[0]]=[raw[1]]
            sop_text_name[raw[0]]={raw[2]:raw[1]}
        else:
            sop_true[raw[0]].extend([raw[1]])
            sop_text_name[raw[0]].update({raw[2]:raw[1]})
    for raw in nlu_predict_true+nlu_predict_false:
        if raw[0] not in sop_nlu_predict:
            sop_nlu_predict[raw[0]]=[raw[1]]
        else:
            sop_nlu_predict[raw[0]].extend([raw[1]])
    return sop_true,sop_nlu_predict,sop_text_name
def context_nlu_prf(sop_predict,sop_true):
    context_true=[]
    context_pred=[]
    sop_list=[]
    for raw in sop_true:
        sop_list.extend(sop_true[raw])
    sop_list=list(set([str(raw) for raw in sop_list]))
    if "None" not in sop_list:
        sop_list+=["None"]
    # print(sop_list)
    for raw in sop_true:
        sop_tru=list(set(sop_true[raw]))
        if len(sop_tru)>1:
            sop_tru=[raw for raw in sop_tru if raw!=None]
        if raw in sop_predict:
            sop_pre=list(set(sop_predict[raw])) if sop_predict[raw]!=[] else [None]
        else:
            sop_pre=[None]
        if sop_pre !=[None] and sop_pre==sop_tru:
            for sop_t in sop_tru:
                context_true.extend([sop_list.index(sop_t)])
                context_pred.extend([sop_list.index(sop_t)])
        else:
            sop_max=sop_pre if len(sop_pre) >= len(sop_tru) else sop_tru
            for sop_index in range(len(sop_max)):
                if sop_index < len(sop_tru):
                    if sop_tru[sop_index] in sop_list:
                        context_true.extend([sop_list.index(sop_tru[sop_index])])
                    else:
                        context_true.extend([sop_list.index("None")])
                else:
                    context_true.extend([sop_list.index("None")])
                if sop_index < len(sop_pre):
                    if sop_pre[sop_index] in sop_list:
                        context_pred.extend([sop_list.index(sop_pre[sop_index])])
                    else:
                        context_pred.extend([sop_list.index("None")])
                else:
                    context_pred.extend([sop_list.index("None")])
    assert len(context_true) == len(context_pred),"len(context_true):%s ,len(context_pred):%s "%(len(context_true),len(context_pred))
    # print(context_true)
    # print(context_pred)
    f1=classification_report(context_true, context_pred,target_names=sop_list ,digits=4)

    return f1
def context_nlu_key_metric(sop_text_name,context_key_sop,sop_true):
    sop_list=[]
    for raw in sop_true:
        sop_list.extend(sop_true[raw])
    sop_list=list(set([str(raw) for raw in sop_list]))
    key_true_list=[]
    key_pred_list=[]
    for raw in sop_text_name:
        if raw in context_key_sop:
            for sop_text in sop_text_name[raw]:
                key_true_list.append(sop_list.index(str(sop_text_name[raw][sop_text])))
                if sop_text in context_key_sop[raw]:
                    key_pred_list.append(sop_list.index(str(context_key_sop[raw][sop_text])))
                else:
                    key_pred_list.append(sop_list.index("None"))
        else:
            for sop_text in sop_text_name[raw]:
                key_true_list.append(sop_list.index(str(sop_text_name[raw][sop_text])))
                key_pred_list.append(sop_list.index("None"))
    f1=classification_report(key_true_list, key_pred_list,target_names=sop_list ,digits=4)

    return f1
if __name__ == '__main__':
    model = SimcseModel(pretrained_model=pretrain_model, pooling="cls",label_num=8928)
    model.to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    path_or_list=["珑珠积分补录","C2停车权益","订单未发货","权益变更","珑珠券","手机号修改流程","删除实名认证信息","账户无法登录","收不到验证码","C1签约未返珑珠","C1推荐未返珑珠","C3签约未返珑珠"
                  ,"C3推荐未返珑珠","C5签约未返珑珠","C5推荐未返珑珠","C1/C4活动返珑珠","C4缴物业费返珑珠","话费充值","天猫超市卡充值","京东企业购（退换货）","珑珠优选（换货）","珑珠优选（退货）"]
    primary_data,primary_vec,similar_data=make_primary_vec(path_or_list,model)
    threshold=0.8
    predict_save_path="./context_nlu_pred.txt"
    sop_true,sop_nlu_predict,sop_text_name=read_true_sop("推荐SOP数据804-811-手动进入.xlsx")

    sop_predict,sop_top_predict,context_key_sop=dialogue_pridect("./sop_test_0825_data.xlsx",model,primary_data,primary_vec,threshold,similar_data,predict_save_path,sop_text_name,topN=5)

    # print(sop_true)
    bad_context_id=context_nlu_metric(sop_predict,sop_true)
    # print(bad_context_id)
    bad_context_id_top=context_nlu_metric(sop_top_predict,sop_true)
    for raw in sop_predict:
        if set(sop_predict[raw])!=set(sop_top_predict[raw]):
            print(raw,set(sop_predict[raw]),set(sop_top_predict[raw]))

    f1=context_nlu_prf(sop_predict,sop_true)
    print("dialogue:")
    print(f1)
    f1_topN=context_nlu_prf(sop_top_predict,sop_true)
    print("dialogue Top5:")
    print(f1_topN)
    f1=context_nlu_key_metric(sop_text_name,context_key_sop,sop_true)
    print("dialogue key sentence:")
    print(f1)

    # nlu_bad_context_id=nlu_metric(sop_nlu_predict,sop_true)

    # print(nlu_bad_context_id)


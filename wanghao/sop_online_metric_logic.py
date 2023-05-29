#!/usr/bin/python3  
# -*- coding: utf-8 -*-
# @Time    : 2022/8/17 9:58
# @Author  : wanghao27
# @Site    : 
# @File    : sop_metric_data.py
# @Email   : wanghao27@longfor.com
import json
import re

import requests
import torch

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm
import torch.nn.functional as F

from clean import clean
from sop_metric_logic import context_nlu_key_metric
from utils.oss_cfg import OSS2
from typing import List


BATCH_SIZE = 64
LR = 2e-5
MAXLEN = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nlu_emb_url="https://api.longhu.net/chatbot-nlu-update-index-prod/embedding"
nlu_emb_header= {
    'Content-Type': "application/json",
    'X-Gaia-Api-Key':"8e21002e-1064-417d-af10-7ac1c4e5601f"
}
predict_dialogue_save_path="newTest_allknow_basecode.xlsx"


def make_primary_vec(path_or_list):
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
    out_key_data=[]
    for raw in primary_data:
        print(raw)
        if raw in primary_q:
            primary_index=primary_q.index(raw)
            similar_list=similar_q[primary_index].split("###")
            out_primary_data.extend([raw]*(len(similar_list)+1))
            similar_data.extend([raw]+similar_list)
            out_key_data.extend([raw]+similar_list)
        else:
            similar_list=[]
            out_key_data.extend([raw])
            out_primary_data.extend([raw])
        primary_vec=torch.cat((primary_vec,model_pridect([raw]+similar_list)),dim=0)
    assert primary_vec.size()[0]==len(out_primary_data)
    assert primary_vec.size()[0]==len(out_key_data)
    saved_json={}
    primary_list=primary_vec.tolist()
    with open("primary_simlary_embedding.json","w",encoding="utf-8") as f :
        for i,raw in enumerate(out_key_data):
            saved_json[raw]=[out_primary_data[i],primary_list[i]]
        json.dump(saved_json,f,ensure_ascii=False,indent=4)
    return out_primary_data,primary_vec,similar_data


def model_pridect(text):
    all_data_vector=torch.tensor([])
    for raw in text:
        nlu_inputs={"text":raw}
        nlu_inputs=json.dumps(nlu_inputs)
        data=requests.post(nlu_emb_url, data=nlu_inputs.encode("utf-8"),headers=nlu_emb_header).json()
        emb=torch.tensor([data["data"]])
        all_data_vector=torch.cat((all_data_vector,emb),dim=0)
    return all_data_vector

def dialogue_pridect(path,primary_data,primary_vec,thres,similar_data,predict_save_path,predict_vec_path,sop_text_name,sop_list,topN=5):
    sop_test_data=pd.read_excel(path,keep_default_na=False)
    context_id=sop_test_data["context_id"]
    context_data=sop_test_data["context_data"]
    context_label=sop_test_data["context_label"]
    context_pridect=[]
    context_pridect_score=[]
    context_sop={}
    context_top_sop={}
    curr_context_id=""
    context_key_sop={}
    with open(predict_vec_path,"r",encoding="utf-8") as f:
        vec_predict=json.load(f)
    with open(predict_save_path,"w",encoding="utf-8") as f:
        for i,raw in enumerate(tqdm(context_data)):
            print(context_id[i]!="")
            if context_id[i]!="":
                curr_context_id=context_id[i]
                context_sop[curr_context_id]=[]
                context_top_sop[curr_context_id]=[]
                context_intent_change=[]
                if "_begin" not in raw and "None" not in raw:
                    # idata=re.sub("[http|https]://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+","",raw)
                    # idata=re.sub("(<img).*(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+","<img",idata)
                    idata=clean(raw)
                    dialogue_input_data=[idata]
                else:
                    dialogue_input_data=[]
            else:
                if "_begin" not in raw and "None" not in raw:
                    # idata=re.sub("[http|https]://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+","",raw)
                    # idata=re.sub("(<img).*(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+","<img",idata)
                    # print(idata)
                    idata=clean(raw)
                    dialogue_input_data.extend([idata])
                else:
                    context_pridect.append(None)
                    context_pridect_score.append(None)
                    continue
            if len(dialogue_input_data)>0:
                # dataloader = DataLoader(PriDataset([dialogue_input_data[-1]]), batch_size=BATCH_SIZE,collate_fn=collate_fn)
                # dataloader = DataLoader(PriDataset(["/".join(dialogue_input_data[-5:])]), batch_size=BATCH_SIZE,collate_fn=collate_fn)
                # print(dialogue_input_data[-1])
                one_vector=torch.tensor(vec_predict[dialogue_input_data[-1]]).to(DEVICE)
                sop_cos=F.cosine_similarity(one_vector,primary_vec)
                sop_cos_argmax=sop_cos.argmax()
                # print(sop_cos_argmax)
                sop_cos_max=sop_cos.max()
                sop_hit=[(sop_cos[i],raw) for i,raw in enumerate(primary_data) if raw in sop_list and sop_cos[i]>thres]
                sop_top_hit_cos=0
                sop_top_hit_name=None
                if len(sop_hit)>0:
                    for raw in sop_hit:
                        if raw[0]>sop_top_hit_cos:
                            sop_top_hit_cos=raw[0].cpu()
                            sop_top_hit_name=raw[1]
                    if len(context_intent_change)>0:
                        if context_intent_change[-1]!=sop_top_hit_name:
                            context_intent_change.append(sop_top_hit_name)
                            context_pridect.append(sop_top_hit_name)
                            context_pridect_score.append(sop_top_hit_cos)
                            context_sop[curr_context_id].append(sop_top_hit_name)
                            if len(dialogue_input_data)<topN:
                                context_top_sop[curr_context_id].append(sop_top_hit_name)
                        else:
                            context_pridect.append(None)
                            context_pridect_score.append(None)
                    else:
                        context_intent_change.append(sop_top_hit_name)
                        context_pridect.append(sop_top_hit_name)
                        context_pridect_score.append(sop_top_hit_cos)
                        context_sop[curr_context_id].append(sop_top_hit_name)
                        if len(dialogue_input_data)<topN:
                            context_top_sop[curr_context_id].append(sop_top_hit_name)
                    continue
                else:
                    sop_top_hit_name=None
                if sop_cos_max>thres and sop_top_hit_name==None:
                    if len(context_intent_change)>0:
                        if context_intent_change[-1]!=primary_data[sop_cos_argmax]:
                            context_intent_change.append(primary_data[sop_cos_argmax])
                            context_pridect.append(primary_data[sop_cos_argmax])
                            context_pridect_score.append(sop_cos_max.cpu())
                            context_sop[curr_context_id].append(primary_data[sop_cos_argmax])
                            if len(dialogue_input_data)<topN:
                                context_top_sop[curr_context_id].append(primary_data[sop_cos_argmax])
                        else:
                            context_pridect.append(None)
                            context_pridect_score.append(None)
                    else:
                        context_intent_change.append(primary_data[sop_cos_argmax])
                        context_pridect.append(primary_data[sop_cos_argmax])
                        context_pridect_score.append(sop_cos_max.cpu())
                        context_sop[curr_context_id].append(primary_data[sop_cos_argmax])
                        if len(dialogue_input_data)<topN:
                            context_top_sop[curr_context_id].append(primary_data[sop_cos_argmax])
                    print(primary_data[sop_cos_argmax],":",sop_cos_max.cpu(),":","/".join(dialogue_input_data[-1]),":",similar_data[sop_cos_argmax])
                    f.write(primary_data[sop_cos_argmax]+":"+str(sop_cos_max.cpu().item())+":"+"/".join(dialogue_input_data[-1])+":"+similar_data[sop_cos_argmax]+"\n")
                else:
                    context_pridect.append(None)
                    context_pridect_score.append(None)
            else:
                context_pridect.append(None)
                context_pridect_score.append(None)
                # print(dialogue_input_data)
    assert len(context_pridect)==len(context_id),"context_pridect_len:%s,context_id_len:%s"%(len(context_pridect),len(context_id))

    pridect_data=pd.DataFrame({"context_id":context_id,"context_data":context_data,"context_label":context_label,"context_predict":context_pridect,"predict_score":context_pridect_score})

    pridect_data.to_excel(predict_dialogue_save_path)

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
            # print(sop_predict[raw])
            sop_pre=list(set(sop_predict[raw])) if sop_predict[raw]!=[] else [None]
        else:
            sop_pre=[None]
        if sop_tru==sop_pre:
            true_num+=1
        else:
            bad_context_id[raw]={"true":sop_tru,"pred":sop_pre}
    print(true_num)
    print("context_nlu_acc:%s"%str(true_num/all_true))
    return bad_context_id

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
            sop_tru=list(set([raw for raw in sop_tru if raw!=None]))
        if raw in sop_predict:
            sop_pre=list(set(sop_predict[raw])) if sop_predict[raw]!=[] else [None]
        else:
            sop_pre=[None]
        if sop_pre !=[None] and sop_pre==sop_tru:
            for sop_t in sop_tru:
                context_true.extend([sop_list.index(sop_t)])
                context_pred.extend([sop_list.index(sop_t)])
        else:
            sop_pre_tru_com=[raw for raw in sop_pre if raw in sop_tru]
            sop_pre_not_com=[raw for raw in sop_pre if raw not in sop_pre_tru_com]
            sop_pre_not_com.sort()
            sop_tru_not_com=[raw for raw in sop_tru if raw not in sop_pre_tru_com]
            sop_tru_not_com.sort()
            sop_pre=sop_pre_tru_com+sop_pre_not_com
            sop_tru=sop_pre_tru_com+sop_tru_not_com
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
def nlu_metric(nlu_predict,sop_true):
    all_true=len(list(sop_true.keys()))
    true_num=0
    bad_context_id={}
    for raw in sop_true:
        sop_tru=list(set(sop_true[raw]))
        if len(sop_tru)>1:
            sop_tru=list(set([iraw for iraw in sop_tru if iraw!=None]))
        if raw in nlu_predict:
            sop_pre=list(set(nlu_predict[raw])) if nlu_predict[raw]!=[] else [None]
        else:
            sop_pre=list(set([None]))
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

def make_dialogue_vec(path_or_list):
    sop_test_data=pd.read_excel(path_or_list,keep_default_na=False)
    context_data=sop_test_data["context_data"]
    dialogue_save_vec={}
    good_data=[]
    for i,raw in enumerate(tqdm(context_data)):
        if "_begin" not in raw and "None" not in raw:
            # idata=re.sub("[http|https]://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+","",raw)
            # idata=re.sub("(<img).*(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+","<img",idata)
            idata=clean(raw)
            good_data.extend([idata])
        else:
            continue
    dialogue_vec=model_pridect(good_data)
    dialogue_list=dialogue_vec.tolist()
    for i,raw in enumerate(good_data):
        dialogue_save_vec[raw]=dialogue_list[i]
    with open("dialogue_embedding_0902.json","w",encoding="utf-8") as f:
        json.dump(dialogue_save_vec,f,ensure_ascii=False,indent=4)
    return dialogue_save_vec
if __name__ == '__main__':
    # vec=model_pridect(["你好"])
    # print(vec.size())
    sop_list=["珑珠积分补录","C2停车权益","订单未发货","权益变更","珑珠券","手机号修改流程","删除实名认证信息","账户无法登录","收不到验证码","C1签约未返珑珠","C1推荐未返珑珠","C3签约未返珑珠"
        ,"C3推荐未返珑珠","C5签约未返珑珠","C5推荐未返珑珠","活动返珑珠","C4缴物业费返珑珠","话费充值","天猫超市卡充值","京东企业购（退换货）","珑珠优选（换货）","珑珠优选（退货）"]
    # primary_data,primary_vec,similar_data=make_primary_vec(path_or_list)
    # dialogue_save_vec=make_dialogue_vec("./all_test_0913_data.xlsx")
    # print(primary_vec.size())
    with open("know_rec_primary_simlary_embedding_0919.json","r",encoding="utf-8") as f:
        primary_json=json.load(f)
    similar_data=[]
    primary_data=[]
    all_know=[]
    primary_vec=torch.tensor([]).to(DEVICE)
    know_rec=pd.read_excel("knowledge_basecode_data_pro.xlsx")
    # primary_q=list([raw.replace(" ","") for raw in know_rec["primary_question"]])
    primary_q=list(know_rec["primary_question"])
    similar_q=list(know_rec["similar_question"])
    # all_know.extend(primary_q)
    # for raw in similar_q:
    #     if isinstance(raw,str):
    #         print(raw)
    #         all_know=all_know+raw.split("###")
    # all_know=list(set(all_know))
    # primary_json={}
    # for raw in tqdm(all_know):
    #     know_vec=model_pridect([raw])
    #     primary_json[raw]=know_vec.tolist()[0]
    # with open("know_rec_primary_simlary_embedding_0919.json","w",encoding="utf-8") as f :
    #     json.dump(primary_json,f,ensure_ascii=False,indent=4)

    for i,raw in enumerate(primary_q):
        if isinstance(similar_q[i],str):
            print(similar_q[i])
            similar_list=similar_q[i].split("###")
            primary_data.extend([raw]*(len(similar_list)+1))
            similar_data.extend([raw]+similar_list)
        else:
            similar_data.extend([raw])
            primary_data.extend([raw])

    print(len(similar_data),len(primary_data))
    for raw in similar_data:
        # similar_data.extend([raw])
        # primary_data.extend([primary_json[raw][0]])
        primary_vec=torch.cat((primary_vec,torch.tensor([primary_json[raw]]).to(DEVICE)),dim=0)
        # primary_vec=torch.cat((primary_vec,torch.tensor([primary_json[raw.replace(" ","")][1]]).to(DEVICE)),dim=0)
    threshold=0.8
    predict_save_path="./context_nlu_pred.txt"
    sop_true,sop_nlu_predict,sop_text_name=read_true_sop("SOP和知识数据标注文档(知识推荐数据-0902-516条-未审核).xlsx")

    sop_predict,sop_top_predict,context_key_sop=dialogue_pridect("./all_test_0913_data.xlsx",primary_data,primary_vec,threshold,similar_data,predict_save_path,"dialogue_embedding_0913.json",sop_text_name,sop_list,topN=5)
    #

    print(sop_true)
    bad_context_id=context_nlu_metric(sop_predict,sop_true)
    # print(bad_context_id)
    bad_context_id_top=context_nlu_metric(sop_top_predict,sop_true)

    f1=context_nlu_prf(sop_predict,sop_true)
    print("dialogue:")
    print(f1)

    f1_topN=context_nlu_prf(sop_top_predict,sop_true)
    print("dialogue Top5:")
    print(f1_topN)

    sop_true=list(set(primary_data))
    f1=context_nlu_key_metric(sop_text_name,context_key_sop,sop_true)
    print("dialogue key sentence:")
    print(f1)


    # nlu_bad_context_id=nlu_metric(sop_nlu_predict,sop_true)
    #
    # print(nlu_bad_context_id)


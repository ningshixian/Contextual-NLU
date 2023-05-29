#!/usr/bin/python3  
# -*- coding: utf-8 -*-
# @Time    : 2022/8/10 16:21
# @Author  : wanghao27
# @Site    : 
# @File    : history_acc_metric.py
# @Email   : wanghao27@longfor.com
import pandas as pd
import torch
from sklearn import metrics
from sklearn.metrics import classification_report

def read_excel_1(path,sheet_name):
    data=pd.read_excel(path,sheet_name=sheet_name)
    if sheet_name=="20220808手工进入SOP数据":
        sop_activte_true=[1 for raw in data["是否应使用SOP"] if raw=="是"]
        sop_activate_pred=len(sop_activte_true)*[0]
    elif sheet_name=="推荐进入SOP数据":
        sop_activte_true=[1 if raw=="是" else 0 for raw in data["是否应使用SOP"] if raw in ["是","否"]]
        sop_activate_pred=len(sop_activte_true)*[1]
    return sop_activte_true,sop_activate_pred
def read_excel_2(path,sheet_name):
    data=pd.read_excel(path,sheet_name=sheet_name)
    sop_activte_true=[1 if raw=="是" else 0 for raw in data["重新标注"] if raw in ["是","否"]]
    sop_activate_pred=len(sop_activte_true)*[1]
    return sop_activte_true,sop_activate_pred
if __name__ == '__main__':
    sop_true=[]
    sop_pred=[]
    true,pred=read_excel_1("推荐SOP及手工进入SOP数据.xlsx","20220808手工进入SOP数据")
    sop_true.extend(true)
    sop_pred.extend(pred)
    true,pred=read_excel_2("推荐SOP数据804-811(标注完).xlsx","Sheet1")
    acc=torch.sum((torch.tensor(true)==torch.tensor(pred)))
    print(acc/len(true))
    sop_true.extend(true)
    sop_pred.extend(pred)
    acc=torch.sum((torch.tensor(sop_true)==torch.tensor(sop_pred)))
    acc=acc/len(sop_true)
    print(acc)
    acc=metrics.accuracy_score( torch.tensor(sop_true),torch.tensor(sop_pred))
    print(acc)
    f1=classification_report(sop_true, sop_pred,labels=[0,1] ,digits=4,output_dict=True)
    print(f1)

#!/usr/bin/python3  
# -*- coding: utf-8 -*-
# @Time    : 2022/8/17 9:58
# @Author  : wanghao27
# @Site    : 
# @File    : sop_metric_data.py
# @Email   : wanghao27@longfor.com
import argparse
import re
import pandas as pd
import pymysql

from utils import ApolloCfg
from utils.oss_cfg import OSS2


def get_dialogue_input(host, user, password, database,table_name,context_id):
    connection = pymysql.connect(host=host, user=user, password=password,port=3306,db=database,charset='utf8',autocommit=True)
    context_data={}
    try:
        with connection.cursor() as cursor:
            for i,id in enumerate(context_id):
                if isinstance(id,str):
                    request_sql="select create_time,user_input,response,sop_code from `%s`.`%s` where context_id='%s' and call_type='in' order by create_time asc "%(database,table_name,id)
                    cursor.execute(request_sql)
                    data_all=cursor.fetchall()
                    for i,raw in enumerate(data_all):
                        text = re.sub("(?<!\d)(\d{18})(?!\d)","***",str(raw[1])) #去除身份证号
                        text = re.sub("(?<!\d)(1\d{10})(?!\d)","***",str(text)) #去除手机号码
                        if id in context_data:
                            context_data[id].append(text)
                        else:
                            context_data[id]=[text]
                    connection.commit()
    except Exception as e:
        print(e)
    finally:
        connection.close()
        return context_data

def read_test_datasets(path):
    data=pd.read_excel(path)
    context_id=data["会话ID（context_id）"]
    sop_input=data["触发文本"]
    sop_name=data["触发SOP名称"]
    sop_label=data["分析触发文本及以上会话内容，判断推荐SOP是否准确"]
    return context_id,sop_input,sop_name,sop_label

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="test")
    args = parser.parse_args()
    if args.env=="test":
        apollo=ApolloCfg("test",decrypt_fields=["db1.pass_word"],namespace=["wanghao27_other","wanghao27_config"])
    else:
        apollo=ApolloCfg("pro",decrypt_fields=["db1.pass_word"],namespace=["wanghao27_other","wanghao27_config"])
    other_config=apollo.config["wanghao27_other"]
    nlp_config=apollo.config["wanghao27_config"]

    host, user, password, database,table_name=other_config["db1.address"],other_config["db1.user"],other_config["db1.pass_word"],other_config["db1.name"],"t_seat_assist_dialogue_log"
    test_dataset_path="推荐SOP数据804-811-手动进入.xlsx"
    context_id,sop_input,sop_name,sop_label=read_test_datasets(test_dataset_path)
    context_data=get_dialogue_input(host, user, password, database,table_name,set(context_id))
    test_data_label={}
    for i,raw in enumerate(context_id):
        try:
            sop_index=context_data[raw].index(sop_input[i])
        except Exception as e:
            sop_index=0
        if raw not in test_data_label:
            print(raw)
            label_init=[None]*len(context_data[raw])
            label_init[sop_index]=sop_name[i]
            test_data_label[raw]=label_init
        else:
            test_data_label[raw][sop_index]=sop_name[i]

    assert len(list(context_data.keys()))==len(list(test_data_label.keys()))
    context_id_list=[]
    context_data_list=[]
    context_label_list=[]
    for raw in context_data:
        print(raw)
        context_id_list.extend([raw])
        context_id_list.extend([None]*(len(context_data[raw])-1))
        context_data_list.extend(context_data[raw])
        context_label_list.extend(test_data_label[raw])
    assert len(context_id_list)==len(context_data_list)
    assert len(context_id_list)==len(context_label_list)

    test_df=pd.DataFrame({"context_id":context_id_list,"context_data":context_data_list,"context_label":context_label_list})

    test_df.to_excel("./sop_test_data.xlsx",engine="xlsxwriter")

    # oss_file=OSS2(nlp_config)
    # true_dataset_url=oss_file.oss_put("datasets/sop_test_0825_data.xlsx","sop_test_data.xlsx")
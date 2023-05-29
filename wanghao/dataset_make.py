#!/usr/bin/python3  
# -*- coding: utf-8 -*-
# @Time    : 2022/8/10 17:40
# @Author  : wanghao27
# @Site    : 
# @File    : dataset_make.py
# @Email   : wanghao27@longfor.com
import argparse
import json
import re

import pymysql

from utils import ApolloCfg
from utils.oss_cfg import OSS2


def read_sop_dialogue_log(host, user, password, database,table_name):
    connection = pymysql.connect(host=host, user=user, password=password,port=3306,db=database,charset='utf8',autocommit=True)
    try:
        with connection.cursor() as cursor:
            request_sql="select context_id from `{0}`.`{1}` where source='sop'and user_input = '_begin' and response_code = 0 " \
                        "and original_bot_code in ('LZZXKF','C2ZXKF','LZZXKFWBYH','DHYQY','DHYTYDL','LZYX',  'TEZSXTTOC', 'C3GYKF')" \
                        "and seat_account not in ('wanghaoyu', 'a-yuxiaotong', 'lihuiwu','w_zhangrongfei2','undefined','bashan','sunchao01','linjiling','a-liping','a-zhangxiaoping','w_machengyao')" \
                        "and DATE_FORMAT(create_time,'%Y-%m-%d') > date_sub(curdate(),interval 1 month)".format(database,table_name)
            cursor.execute(request_sql)
            data_all=cursor.fetchall()
            sop_context=[raw[0] for raw in data_all]
            connection.commit()
    except Exception as e:
        print(e)
    finally:
        connection.close()
        return sop_context

def read_sop_all_dialogue_log(host, user, password, database,table_name):
    connection = pymysql.connect(host=host, user=user, password=password,port=3306,db=database,charset='utf8',autocommit=True)
    try:
        with connection.cursor() as cursor:
            request_sql="select context_id from `{0}`.`{1}` where response_code = 0 " \
                        "and original_bot_code in ('LZZXKF','C2ZXKF','LZZXKFWBYH','DHYQY','DHYTYDL','LZYX',  'TEZSXTTOC', 'C3GYKF')" \
                        "and seat_account not in ('wanghaoyu', 'a-yuxiaotong', 'lihuiwu','w_zhangrongfei2','undefined','bashan','sunchao01','linjiling','a-liping','a-zhangxiaoping','w_machengyao')" \
                        "and DATE_FORMAT(create_time,'%Y-%m-%d') > date_sub(curdate(),interval 1 month)".format(database,table_name)
            cursor.execute(request_sql)
            data_all=cursor.fetchall()
            all_context=[raw[0] for raw in data_all]
            connection.commit()
    except Exception as e:
        print(e)
    finally:
        connection.close()
        return all_context

def read_sop_input_false_dialogue_log(host, user, password, database,table_name,context_id,context_id_true):
    connection = pymysql.connect(host=host, user=user, password=password,port=3306,db=database,charset='utf8',autocommit=True)
    context_data={}
    context_label={}
    context_id=[raw for raw in set(context_id) if raw not in set(context_id_true)]
    try:
        with connection.cursor() as cursor:
            for i,id in enumerate(context_id):
                request_sql="select create_time,user_input,response,sop_code from `%s`.`%s` where context_id='%s' order by create_time asc "%(database,table_name,id)
                cursor.execute(request_sql)
                data_all=cursor.fetchall()
                for i,raw in enumerate(data_all):
                    text = re.sub("(?<!\d)(\d{18})(?!\d)","***",str(raw[1])) #去除身份证号
                    text = re.sub("(?<!\d)(1\d{10})(?!\d)","***",str(text)) #去除手机号码
                    if id in context_data:
                        context_data[id].append(text)
                    else:
                        context_data[id]=[text]
                context_label[id]=[None]
                connection.commit()
    except Exception as e:
        print(e)
    finally:
        connection.close()
        return context_data,context_label

def read_sop_input_dialogue_log(host, user, password, database,table_name,context_id):
    connection = pymysql.connect(host=host, user=user, password=password,port=3306,db=database,charset='utf8',autocommit=True)
    context_data={}
    context_label={}
    context_id=set(context_id)
    try:
        with connection.cursor() as cursor:
            # print(context_id)
            for i,id in enumerate(context_id):
                request_sql="select create_time,user_input,response,sop_code from `%s`.`%s` where context_id='%s' order by create_time asc "%(database,table_name,id)
                # print(request_sql)
                cursor.execute(request_sql)
                data_all=cursor.fetchall()
                for i,raw in enumerate(data_all):
                    print(id,raw[1])
                    if id in context_data:
                        if raw[1] != '_begin' and raw[1]!=None:
                            text = re.sub("(?<!\d)(\d{18})(?!\d)","***",str(raw[1])) #去除身份证号
                            text = re.sub("(?<!\d)(1\d{10})(?!\d)","***",str(text)) #去除手机号码
                            context_data[id].append(text)
                        else:
                            if raw[1]=='_begin':
                                print('_begin')
                                try:
                                    if id in context_label:
                                        if {len(context_data[id])-1:json.loads(raw[2])[0]['intent']} not in context_label[id]:
                                            context_label[id].append({len(context_data[id])-1:[json.loads(raw[2])[0]['intent'],raw[3]]})
                                    else:
                                        context_label[id]=[{len(context_data[id])-1:[json.loads(raw[2])[0]['intent'],raw[3]]}]
                                except Exception as e:
                                    print(e)
                    else:
                        if raw[1] == '_begin' or raw[1]==None:
                            break
                        else:
                            assert raw[1] != '_begin'
                            text = re.sub("(?<!\d)(\d{18})(?!\d)","***",str(raw[1])) #去除身份证号
                            text = re.sub("(?<!\d)(1\d{10})(?!\d)","***",str(text)) #去除手机号码
                            context_data[id]=[text]
                connection.commit()
    except Exception as e:
        print(e)
    finally:
        connection.close()
        return context_data,context_label


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
    sop_context=read_sop_dialogue_log(host, user, password, database,table_name)
    context_data,context_label=read_sop_input_dialogue_log(host, user, password, database,table_name,sop_context)
    sop_all_context=read_sop_all_dialogue_log(host, user, password, database,table_name)
    false_context_data,false_context_label=read_sop_input_false_dialogue_log(host, user, password, database,table_name,sop_all_context,sop_context)
    context_true_all={}
    context_false_all={}
    print(context_data)
    print(context_label)
    print(false_context_data)
    print(false_context_label)
    for raw in context_data:
        if raw in context_data and raw in context_label:
            context_true_all[raw]={"data":context_data[raw],"label":context_label[raw]}
    for raw in false_context_data:
        if raw in false_context_data and raw in false_context_label:
            context_false_all[raw]={"data":false_context_data[raw],"label":false_context_label[raw]}
    with open("datasets_true_%s.json"%args.env,"w",encoding="utf-8") as f :
        json.dump(context_true_all,f,ensure_ascii=False,indent=4)
    with open("datasets_false_%s.json"%args.env,"w",encoding="utf-8") as f :
        json.dump(context_false_all,f,ensure_ascii=False,indent=4)

    oss_file=OSS2(nlp_config)
    true_dataset_url=oss_file.oss_put("datasets/context_nlu_datasets_true_%s.json"%args.env,"./datasets_true_%s.json"%args.env)
    false_dataset_url=oss_file.oss_put("datasets/context_nlu_datasets_false_%s.json"%args.env,"./datasets_false_%s.json"%args.env)
    print(true_dataset_url)
    print(false_dataset_url)
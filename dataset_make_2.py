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
import traceback
from tqdm import tqdm 
import pymysql
from utils.preprocess import clean
from utils import ApolloCfg
from utils.oss_cfg import OSS2
from utils.DataPersistence import *

"""python dataset_make_2.py --env prod
"""


# This function reads the sop dialogue log from the database.
#     :return: the context_id of the sop dialogue log
def get_sop_context_id(host, user, password, database,table_name):
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

def get_zuoxifuzhu_context_id(host, user, password, database,table_name):
    connection = pymysql.connect(host=host, user=user, password=password,port=3306,db=database,charset='utf8',autocommit=True)
    try:
        with connection.cursor() as cursor:
            request_sql="select context_id from `{0}`.`{1}` " \
                        "WHERE source='zuoxifuzhu' and res_mode=1 and knowledge_id <> 10086 AND user_input <> '转人工' " \
                        "and (edited = 1 or sended = 1) " \
                        "and seat_account not in ('wanghaoyu', 'a-yuxiaotong', 'lihuiwu','w_zhangrongfei2','undefined','bashan','sunchao01','linjiling','a-liping','a-zhangxiaoping','w_machengyao') " \
                        "and original_bot_code in ('LZZXKF','C2ZXKF','LZZXKFWBYH','DHYQY','DHYTYDL','LZYX',  'TEZSXTTOC', 'C3GYKF') " \
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

# Returns a list of all context_id in the table.
#     :param host: host of the database
#     :param user: user of the database
#     :param password: password of the database
#     :param database: database name
#     :param table_name: table name
#     :return: a list of all context_id in the table
def get_all_context_id(host, user, password, database,table_name):
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


def build_negative_examples(host, user, password, database,table_name,context_id,context_id_true):
    connection = pymysql.connect(host=host, user=user, password=password,port=3306,db=database,charset='utf8',autocommit=True)
    context_data={}
    context_label={}
    context_id=[raw for raw in set(context_id) if raw not in set(context_id_true)]
    try:
        with connection.cursor() as cursor:
            for i,id in enumerate(context_id):
                request_sql = """
                    select create_time,user_input,response,sop_code
                    from `%s`.`%s` 
                    where context_id='%s'
                    order by create_time asc 
                    """ % (database, table_name, id)
                cursor.execute(request_sql)
                data_all=cursor.fetchall()
                for i,raw in enumerate(data_all):
                    text = clean(str(raw[1]))
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

def build_positive_examples(host, user, password, database,table_name,context_id_true):
    connection = pymysql.connect(host=host, user=user, password=password,port=3306,db=database,charset='utf8',autocommit=True)
    context_data={}
    context_label={}
    # context_id_true=set(context_id_true)

    with connection.cursor() as cursor:
        for i,id in tqdm(enumerate(context_id_true)):
            try:
                # request_sql = """
                #     select create_time,message,user_input,response,sop_code,source,edited,sended 
                #     from `%s`.`%s` 
                #     where context_id='%s'
                #     order by create_time asc 
                #     """ % (database, table_name, id)
                request_sql = """
                    SELECT b.createtime, b.message, c.user_input, c.response
                        , c.sop_code, c.source, c.edited, c.sended
                    FROM (
                        SELECT *
                        FROM oc_chat_message a
                        WHERE a.contextid = '%s'
                            AND tousername <> '龙小湖'
                            AND username <> '龙小湖'
                            AND calltype = 'in'
                    ) b
                        LEFT JOIN (
                            SELECT *
                            FROM `%s`.`%s`
                            WHERE context_id = '%s'
                        ) c
                        ON b.message_id = c.message_id
                    ORDER BY createtime ASC
                """ % (id, database, table_name, id)
                cursor.execute(request_sql)
                data_all=cursor.fetchall()
                pre_label = None
                for i,row in enumerate(data_all):
                    #当 source=sop，user_input=_begin；否则同 message
                    _, message, user_input, response, sop_code, source, edited, sended = row
                    # 筛掉输入为空以及来源为自动邀评、质检、预警的 utterance
                    if message==None or (source and source not in ('sop', 'zuoxifuzhu')): 
                        continue
                    # 筛掉回复体异常的 utterance
                    if response and response != '[]':
                        try:
                            intent = json.loads(response)[0]['intent']
                            intent_id = json.loads(response)[0]['id']
                        except Exception as e:
                            print(e)
                            continue
                    
                    if source=="sop" and user_input=='_begin' and id not in context_data: 
                        # 如果对话的第一句为'_begin'，结束循环
                        break
                    elif source=="sop" and user_input=='_begin':  # 开始进入 SOP 流程
                        context_label.setdefault(id, [])
                        idx = len(context_data[id])-1
                        if idx != pre_label:    # 避免重复标签
                            context_label[id].append({idx:[intent,intent_id,sop_code]})
                    elif source=="zuoxifuzhu" and (edited==1 or sended==1):    # 点编辑 or 点发送 (针对当前句子)
                        text = clean(str(message))
                        context_data.setdefault(id, [])
                        context_data[id].append(text)
                        context_label.setdefault(id, [])
                        idx = len(context_data[id])-1
                        context_label[id].append({idx:[intent,intent_id,sop_code]})
                        pre_label = idx     # 记录当前句子所在的数组下标
                    else:
                        text = clean(str(message))
                        context_data.setdefault(id, [])
                        context_data[id].append(text)

                connection.commit()
            except Exception as e:
                traceback.print_exc()

    connection.close()
    return context_data,context_label


# def build_positive_examples(host, user, password, database,table_name,context_id_true):
#     connection = pymysql.connect(host=host, user=user, password=password,port=3306,db=database,charset='utf8',autocommit=True)
#     context_data={}
#     context_label={}
#     # context_id_true=set(context_id_true)

#     with connection.cursor() as cursor:
#         for i,id in tqdm(enumerate(context_id_true)):
#             try:
#                 request_sql = """
#                     SELECT b.createtime, b.calltype, b.message, c.user_input, c.response
#                         , c.sop_code, c.source, c.edited, c.sended
#                     FROM (
#                         SELECT *
#                         FROM oc_chat_message a
#                         WHERE a.contextid = '%s'
#                             AND tousername <> '龙小湖'
#                             AND username <> '龙小湖'
#                     ) b
#                         LEFT JOIN (
#                             SELECT *
#                             FROM `%s`.`%s`
#                             WHERE context_id = '%s'
#                         ) c
#                         ON b.message_id = c.message_id
#                     ORDER BY createtime ASC
#                 """ % (id, database, table_name, id)
#                 cursor.execute(request_sql)
#                 data_all=cursor.fetchall()
#                 pre_label = None
#                 for i,row in enumerate(data_all):
#                     #当 source=sop，user_input=_begin；否则同 message
#                     _, calltype, message, user_input, response, sop_code, source, edited, sended = row
#                     # 筛掉输入为空以及来源为自动邀评、质检、预警的 utterance
#                     if message==None or (source and source not in ('sop', 'zuoxifuzhu')): 
#                         continue
#                     # 筛掉回复体异常的 utterance
#                     if response and response != '[]':
#                         try:
#                             intent = json.loads(response)[0]['intent']
#                             intent_id = json.loads(response)[0]['id']
#                         except Exception as e:
#                             print(e)
#                             continue
                    
#                     if source=="sop" and user_input=='_begin' and id not in context_data: 
#                         # 如果对话的第一句为'_begin'，结束循环
#                         break
#                     elif source=="sop" and user_input=='_begin':  # 开始进入 SOP 流程
#                         context_label.setdefault(id, [])
#                         idx = len(context_data[id])-1
#                         if idx != pre_label:    # 避免重复标签
#                             context_label[id].append({idx:[intent,intent_id,sop_code]})
#                     elif source=="zuoxifuzhu" and (edited==1 or sended==1):    # 点编辑 or 点发送 (针对当前句子)
#                         text = clean(str(message))
#                         context_data.setdefault(id, [])
#                         context_data[id].append((text, calltype))
#                         context_label.setdefault(id, [])
#                         idx = len(context_data[id])-1
#                         context_label[id].append({idx:[intent,intent_id,sop_code]})
#                         pre_label = idx     # 记录当前句子所在的数组下标
#                     else:
#                         text = clean(str(message))
#                         context_data.setdefault(id, [])
#                         context_data[id].append((text, calltype))

#                 connection.commit()
#             except Exception as e:
#                 traceback.print_exc()

#     connection.close()
#     return context_data,context_label


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="test")
    args = parser.parse_args()
    if args.env == "test":
        apollo = ApolloCfg(
            "test",
            decrypt_fields=["db1.pass_word"],
            namespace=["wanghao27_other", "wanghao27_config"],
        )
    else:
        apollo = ApolloCfg(
            "pro",
            decrypt_fields=["db1.pass_word"],
            namespace=["wanghao27_other", "wanghao27_config"],
        )
    other_config = apollo.config["wanghao27_other"]
    nlp_config = apollo.config["wanghao27_config"]
    host, user, password, database, table_name = (
        other_config["db1.address"],
        other_config["db1.user"],
        other_config["db1.pass_word"],
        other_config["db1.name"],
        "t_seat_assist_dialogue_log",
    )

    # 过滤掉训练集中与测试集重合的 context_id，保证公平
    test_data = pd.read_excel("data/推荐SOP数据804-811-手动进入.xlsx", keep_default_na=False)
    test_data_2 = pd.read_excel("data/知识推荐数据-0902-516条-未审核-知识搜索.xlsx", keep_default_na=False)
    context_id_in_test_set = test_data["会话ID（context_id）"].tolist() + test_data_2["会话ID（context_id）"].tolist()
    
    sop_context = get_sop_context_id(host, user, password, database, table_name)
    zuoxifuzhu_context = get_zuoxifuzhu_context_id(
        host, user, password, database, table_name
    )
    positive_context = list(set(sop_context + zuoxifuzhu_context))
    positive_context = list(filter(lambda x:x not in context_id_in_test_set, positive_context))
    context_data, context_label = build_positive_examples(
        host, user, password, database, table_name, positive_context
    )
    all_context = get_all_context_id(host, user, password, database, table_name)
    all_context = list(filter(lambda x:x not in context_id_in_test_set, all_context))
    false_context_data, false_context_label = build_negative_examples(
        host, user, password, database, table_name, all_context, positive_context
    )

    # print(context_data)
    # print(context_label)
    # print(false_context_data)
    # print(false_context_label)
    context_true_all = {}
    context_false_all = {}
    for raw in context_data:
        if raw in context_data and raw in context_label:
            context_true_all[raw] = {
                "data": context_data[raw],
                "label": context_label[raw],
            }
    for raw in false_context_data:
        if raw in false_context_data and raw in false_context_label:
            context_false_all[raw] = {
                "data": false_context_data[raw],
                "label": false_context_label[raw],
            }
    with open("data/train_datasets_true_%s.json" % args.env, "w", encoding="utf-8") as f:
        json.dump(context_true_all, f, ensure_ascii=False, indent=4)
    with open("data/train_datasets_false_%s.json" % args.env, "w", encoding="utf-8") as f:
        json.dump(context_false_all, f, ensure_ascii=False, indent=4)
    print("写入文件OK!")

    # oss_file=OSS2(nlp_config)
    # true_dataset_url=oss_file.oss_put("datasets/context_nlu_datasets_true_%s.json"%args.env,"./datasets_true_%s.json"%args.env)
    # false_dataset_url=oss_file.oss_put("datasets/context_nlu_datasets_false_%s.json"%args.env,"./datasets_false_%s.json"%args.env)
    # print(true_dataset_url)
    # print(false_dataset_url)
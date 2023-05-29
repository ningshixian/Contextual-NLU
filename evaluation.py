import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from bert4keras.snippets import sequence_padding
from utils.preprocess import clean as clean_seat

"""
sop_true = {
    "context_id": [ "触发SOP名称" ]
}
sop_nlu_predict = {
    "context_id": [ "NLU预测SOP名称" ]
}
sop_true_dict = {
    "context_id": { 触发文本: 触发SOP名称 }
}
sop_predict = {
    "context_id": [ "上下文预测SOP名称" ]
}
context_key_sop: { 
    "context_id": { 触发文本: 上下文预测SOP名称 } 
}
"""

# 分类阈值微调
thredshold = 0.8
max_q_len = 32
training_pkl_path = "model/training_data.pkl"
(num_context, num_classes, D, kid2primary, label2kid) = joblib.load(training_pkl_path)


# 1.Read the sop excel file
# 2. Create a dictionary to store the true sop for each context
# 3. Create a dictionary to store the nlu predicted sop for each context
def read_true_sop(true_path):
    true_sop = pd.read_excel(true_path, keep_default_na=False)  # 会设置缺失值替换为空字符串
    context_id = true_sop["会话ID（context_id）"]
    context_sop_name = true_sop["触发SOP名称"]
    context_sop_text=true_sop["触发文本"]
    context_label = true_sop["重新标注"]
    nlu_predict = true_sop["nlu预测"]
    # context_label = true_sop["分析触发文本及以上会话内容，判断推荐SOP是否准确"]
    # nlu_predict = true_sop["分析触发文本及以上会话内容，判断推荐SOP是否准确"]

    sop_true_dict={}
    sop_true = {}
    for i, raw in enumerate(context_label):
        cid = context_id[i]
        sop_true.setdefault(cid, [])
        sop_true_dict.setdefault(cid, {})
        # 存储每个context的真实 SOP
        sop_true[cid].extend([context_sop_name[i]] if raw == "是" else [None])
        sop_true_dict[cid].update({context_sop_text[i]:context_sop_name[i] if raw == "是" else None})

    sop_nlu_predict = {}
    for i, raw in enumerate(nlu_predict):
        cid = context_id[i]
        sop_nlu_predict.setdefault(cid, [])
        # 存储每个context的 NLU预测 SOP
        sop_nlu_predict[cid].extend([context_sop_name[i]] if raw == "是" else [None])

    return sop_true, sop_nlu_predict, sop_true_dict


# Calculates the accuracy of the NLU model.
def nlu_metric(sop_nlu_predict, sop_true):
    all_true = len(list(sop_true.keys()))
    true_num = 0
    bad_context_id = {}
    for context_id in sop_true:
        res_true=list(set(sop_true[context_id]))
        if len(res_true)>1:
            res_true=[context_id for context_id in res_true if context_id!=None]
        if (context_id in sop_nlu_predict and sop_nlu_predict[context_id]!=[]):
            res_predict = list(set(sop_nlu_predict[context_id]))
        else:
            res_predict = [None]
        # context_id维度：set(标注答案)=set(预测结果) 才算对！
        if res_true == res_predict:
            true_num += 1
        else:
            bad_context_id[context_id] = {"true": res_true, "pred": res_predict}

    print("nlu_acc: %s" % str(true_num / all_true))
    return bad_context_id


# 从对话context维度-计算prf评测指标
def context_nlu_metric(sop_predict, sop_true):
    all_true = len(list(sop_true.keys()))
    true_num = 0
    a,b,c,d = 0,0,0,0
    bad_context_id = {}
    for context_id in sop_true:
        res_true=list(set(sop_true[context_id]))
        if len(res_true)>1:
            res_true=[context_id for context_id in res_true if context_id!=None]
        if (context_id in sop_predict and sop_predict[context_id]!=[]):
            res_predict = list(set(sop_predict[context_id]))
        else:
            res_predict = [None]
        # context_id维度：set(标注答案)=set(预测结果) 才算对！
        if res_true == res_predict:
            true_num += 1
        else:
            bad_context_id[context_id] = {"true": res_true, "pred": res_predict}
        
        if res_predict != [None]:
            a += 1
            if res_true == res_predict: 
                b+=1
        if res_true and res_true != [None]:
            c += 1

    # 从对话context维度-计算prf评测指标
    acc = true_num / all_true
    precision = b/a
    recall = b/c
    f1 = 2*precision*recall / (precision+recall)
    print("推荐正确,推荐错误,推荐总数,标签总数: ", b, a-b, a, c)
    print("acc,p,r,f1: ", acc, precision, recall, f1)
    print("context_nlu_acc: %s" % str(true_num / all_true))
    return None


# 从句子角度计算prf评测指标
def context_nlu_metric_2(context_pridect, context_true):
    a,b,c,d = 0,0,0,0
    bad_context_id = {}
    for i,label in enumerate(context_true):
        predict = context_pridect[i]
        if predict and predict != "None":
            a += 1
            if label and predict == label: 
                b+=1
        if label and label != "None":
            c += 1

    precision = b/a
    recall = b/c
    f1 = 2*precision*recall / (precision+recall)
    print("推荐正确,推荐错误,推荐总数,标签总数: ", b, a-b, a, c)
    print("p,r,f1: ", precision, recall, f1)
    return None


def dialogue_predict(test_set_path, model, tokenizer,sop_true_dict,pri2base,topN=5):
    """根据测试数据上下文进行预测
    """
    sop_test_data = pd.read_excel(test_set_path, keep_default_na=False)  # 会设置缺失值替换为空字符串
    context_id = sop_test_data["context_id"]
    context_data = sop_test_data["context_data"]
    context_label = sop_test_data["context_label"]

    sop_predict = {}
    context_top_sop={}
    context_key_sop={}
    curr_context_id=""
    pre_predict_pri = ""
    context_pridect=[]
    context_pridect_score=[]

    for i,raw in tqdm(enumerate(context_data)):
        if context_id[i]: 
            # 新的对话 → 重置上下文内容 dialogue_input_data
            curr_context_id=context_id[i]
            pre_predict_pri = ""
            sop_predict[curr_context_id] = []
            context_top_sop[curr_context_id]=[]
            if "_begin" not in raw and "None" not in raw:
                raw = clean_seat(raw)
                dialogue_input_data = [raw]
            else:
                dialogue_input_data = []
        else:
            # 否则，附加到后面
            if "_begin" not in raw and "None" not in raw:
                raw = clean_seat(raw)
                dialogue_input_data.extend([raw])
            else:
                context_pridect.append(None)
                context_pridect_score.append(None)
                continue
        
        # 补齐上下文
        if len(dialogue_input_data)<num_context:
            context = [""]*(num_context-len(dialogue_input_data)) + dialogue_input_data
        else:
            context = dialogue_input_data[-num_context:]
        # 模型预测
        token_ids, _ = tokenizer.encode(context[0], maxlen=max_q_len)
        segment_ids = [0] * len(token_ids)
        for sen in context[1:]:
            a,b = tokenizer.encode(sen, maxlen=max_q_len)
            token_ids += a[1:]
            segment_ids += [0] * (len(a)-1)
        probas = model.predict([np.array([token_ids]), np.array([segment_ids])])[0]

        idx = [i for i in range(len(probas)) if probas[i]>thredshold and pri2base[kid2primary[label2kid[i]]]=="ZXSOPBASE"]
        # score>0.8以上的候选，优先推 SOP
        if idx:
            max_idx = max(idx, key=lambda x: probas[x])
            max_score = probas[max_idx]
        else:
            max_score = max(probas)
            max_idx = probas.argmax(axis=0)

        # 相同上下文知识合并为一条
        if pre_predict_pri==max_idx or max_score <= thredshold:
            context_pridect.append(None)
            context_pridect_score.append(None)
        else:   # max_score > thredshold:
            pre_predict_pri = max_idx
            kid = label2kid[max_idx]
            pri = kid2primary[kid]
            sop_predict[curr_context_id].append(pri)
            context_pridect.append(pri)
            context_pridect_score.append(max_score)

            if curr_context_id in sop_true_dict:
                if dialogue_input_data[-1] in list(sop_true_dict[curr_context_id].keys()):
                    if curr_context_id not in context_key_sop:
                        context_key_sop[curr_context_id]={dialogue_input_data[-1]:pri}
                    else:
                        context_key_sop[curr_context_id].update({dialogue_input_data[-1]:pri})

            if len(dialogue_input_data)<topN:
                context_top_sop[curr_context_id].append(pri)

    # pridect_data=pd.DataFrame({"context_id":context_id,"context_data":context_data,"context_label":context_label,"context_predict":context_pridect,"predict_score":context_pridect_score})
    # pridect_data.to_excel("model/sop_context_nlu_predict.xlsx")
    sop_test_data = sop_test_data.drop(labels='Unnamed: 0', axis=1)     # 删掉第一列索引列
    new_right = pd.DataFrame({"context_predict": context_pridect, "predict_score":context_pridect_score})
    df = pd.concat([sop_test_data, new_right], axis=1, join="outer")
    df.to_excel("data/newTest_allknow_result.xlsx",index=False)
    return sop_predict,context_top_sop,context_key_sop


def dialogue_predict_2(test_set_path, model, tokenizer,sop_true_dict,pri2vec,pri2base,topN=5):
    """根据测试数据上下文进行预测
    """
    sop_test_data = pd.read_excel(test_set_path, keep_default_na=False)  # 读取到空字符串时非nan / nrows=500
    context_id = sop_test_data["context_id"]
    context_data = sop_test_data["context_data"]
    context_label = sop_test_data["context_label"]

    all_pri = []
    all_pri_vecs = []
    for k,v in pri2vec.items():
        all_pri_vecs.extend(v)
        all_pri.extend([k]*len(v))

    print("批量预测 test data 上下文向量...")
    dialogue_input_data = []
    context_vecs = []
    batch_token_ids, batch_segment_ids = [], []
    for i,raw in tqdm(enumerate(context_data)):
        if context_id[i]: 
            if batch_token_ids: # 模型预测
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                vecs = model.predict([batch_token_ids, batch_segment_ids])
                context_vecs.extend(vecs)
                # cosine_score = cosine_similarity(all_pri_vecs, vecs)     # [1xxx,768]&[batch,768]→[1xxx,batch]
                # max_idx = np.argmax(cosine_score, axis=1)
                # search_speaker = [[all_pri[id],cosine_score[i][id]] for i, id in enumerate(max_idx)]
                batch_token_ids, batch_segment_ids = [], []
            raw = clean_seat(raw)
            dialogue_input_data = [raw]
        else:
            # 否则，附加到后面
            raw = clean_seat(raw)
            dialogue_input_data.extend([raw])
        context = "[SEP]".join(dialogue_input_data[-num_context:])
        token_ids, segment_ids = tokenizer.encode(context, maxlen=max_q_len)
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
    
    # 遗漏数据补回
    if batch_token_ids:
        batch_token_ids = sequence_padding(batch_token_ids)
        batch_segment_ids = sequence_padding(batch_segment_ids)
        vecs = model.predict([batch_token_ids, batch_segment_ids])
        context_vecs.extend(vecs)

    assert len(context_vecs)==len(context_data)
    # 计算相似度(耗时操作)
    cosine_score_list = cosine_similarity(all_pri_vecs, context_vecs)
    assert len(cosine_score_list)==len(all_pri_vecs)==len(all_pri)
    assert len(cosine_score_list[0])==len(context_data)
    # 取出所有 base=sop 的知识索引
    sop_idx_list = list(filter(lambda i : pri2base[all_pri[i]]=="ZXSOPBASE", range(len(cosine_score_list))))

    print("开始评测（约半个小时）...")
    sop_predict, sop_true = {}, {}
    context_top_sop={}
    context_key_sop={}
    curr_context_id=""
    pre_predict_pri = ""
    context_pridect, context_true = [],[]
    context_pridect_score=[]
    for i,raw in tqdm(enumerate(context_data)):
        if context_id[i]: 
            # 新的对话 → 重置上下文内容 dialogue_input_data
            curr_context_id=context_id[i]
            pre_predict_pri = ""
            sop_predict[curr_context_id] = []
            sop_true[curr_context_id] = []
            context_top_sop[curr_context_id]=[]
            if "_begin" not in raw and "None" not in raw:
                raw = clean_seat(raw)
                dialogue_input_data = [raw]
            else:
                dialogue_input_data = []
        else:
            # 否则，附加到后面
            if "_begin" not in raw and "None" not in raw:
                raw = clean_seat(raw)
                dialogue_input_data.extend([raw])
            else:
                context_pridect.append(None)
                context_pridect_score.append(None)
                continue

        # 取出第i条数据与所有坐席知识的余弦相似度
        cosine_score = list(map(lambda x : x[i], cosine_score_list))
        # score>0.8以上的候选，优先推 SOP
        sop_idx_list_after_filter = list(filter(lambda i : cosine_score[i]>thredshold, sop_idx_list))
        if sop_idx_list_after_filter:
            max_idx = max(sop_idx_list_after_filter, key=lambda j: cosine_score[j])
            max_score = cosine_score[max_idx]
        else:
            max_score = max(cosine_score)
            max_idx = np.argmax(cosine_score)
        
        # if curr_context_id=='caf928fb820b4422903f363ffde7b4d3':
        #     print(raw, pre_predict_pri, max_idx, max_score)
        #     print(pre_predict_pri==max_idx)
        #     print(all_pri[max_idx])
        #     print("\n")
    
        # 后处理策略
        if max_score <= thredshold:
            context_pridect.append(None)
            context_pridect_score.append(None)
        elif pre_predict_pri==all_pri[max_idx]:
            # 相同上下文知识合并为一条
            context_pridect.append(None)
            context_pridect_score.append(None)
        else:   
            # 得分超过阈值，且意图切换了
            pri = all_pri[max_idx]
            pre_predict_pri = pri
            sop_predict[curr_context_id].append(pri)
            context_pridect.append(pri)
            context_pridect_score.append(max_score)

            if curr_context_id in sop_true_dict:
                if dialogue_input_data[-1] in list(sop_true_dict[curr_context_id].keys()):
                    if curr_context_id not in context_key_sop:
                        context_key_sop[curr_context_id]={dialogue_input_data[-1]:pri}
                    else:
                        context_key_sop[curr_context_id].update({dialogue_input_data[-1]:pri})

            if len(dialogue_input_data)<topN:
                context_top_sop[curr_context_id].append(pri)
        
        label = context_label[i]
        context_true.append(label)
        if label: sop_true[curr_context_id].append(label)
    
    # pridect_data=pd.DataFrame({"context_id":context_id,"context_data":context_data,"context_label":context_label,"context_predict":context_pridect,"predict_score":context_pridect_score})
    # pridect_data.to_excel("model/sop_context_nlu_predict_2.xlsx")
    # return sop_predict,context_top_sop,context_key_sop

    # sop_test_data = sop_test_data.drop(labels='Unnamed: 0', axis=1)     # 删掉第一列索引列
    new_right = pd.DataFrame({"context_predict": context_pridect, "predict_score":context_pridect_score})
    df = pd.concat([sop_test_data, new_right], axis=1, join="outer")
    df.to_excel("data/newTest_allknow_result.xlsx",index=False)
    return sop_predict, sop_true, context_pridect, context_true


# def dialogue_predict_3(test_set_path, model, tokenizer,sop_true_dict,topN=5):
#     """根据测试数据上下文进行预测
#     """
#     sop_test_data = pd.read_excel(test_set_path, keep_default_na=False)  # 读取到空字符串时非 nan
#     context_id = sop_test_data["context_id"]
#     context_data = sop_test_data["context_data"]
#     context_label = sop_test_data["context_label"]

#     ZXSOP_knowledge=["珑珠积分补录","C2停车权益","订单未发货","权益变更","珑珠券","手机号修改流程","删除实名认证信息","账户无法登录","收不到验证码","C1签约未返珑珠","C1推荐未返珑珠","C3签约未返珑珠"
#                   ,"C3推荐未返珑珠","C5签约未返珑珠","C5推荐未返珑珠","C1/C4活动返珑珠","C4缴物业费返珑珠","话费充值","天猫超市卡充值","京东企业购（退换货）","珑珠优选（换货）","珑珠优选（退货）"]
#     # 对应的任务描述
#     prefix = u'很相似。'
#     mask_idx = 1
#     pos_id = tokenizer.token_to_id(u'很')
#     neg_id = tokenizer.token_to_id(u'不')

#     sop_predict = {}
#     context_top_sop={}
#     context_key_sop={}
#     curr_context_id=""
#     for i,raw in tqdm(enumerate(context_data)):
#         raw = clean_seat(raw)
#         if context_id[i]: 
#             # 新的对话 → 重置上下文内容 dialogue_input_data
#             curr_context_id=context_id[i]
#             sop_predict[curr_context_id] = []
#             context_top_sop[curr_context_id]=[]
#             if "_begin" not in raw and "None" not in raw:
#                 dialogue_input_data = [raw]
#             else:
#                 dialogue_input_data = []
#         else:
#             # 否则，附加到后面
#             if "_begin" not in raw and "None" not in raw:
#                 dialogue_input_data.extend([raw])
#             else:
#                 continue
        
#         context = dialogue_input_data[-num_context:]
#         context = "[SEP]".join(context)  # 用句号将连续回复拼接起来 '[SEP]' '。'

#         batch_token_ids, batch_segment_ids = [], []
#         for know in ZXSOP_knowledge:
#             text = prefix + context + "[SEP]" + know
#             token_ids, segment_ids = tokenizer.encode(text, maxlen=max_q_len)
#             source_ids, target_ids = token_ids[:], token_ids[:]
#             source_ids[mask_idx] = tokenizer._token_mask_id
#             batch_token_ids.append(source_ids)
#             batch_segment_ids.append(segment_ids)
#         batch_token_ids = sequence_padding(batch_token_ids)
#         batch_segment_ids = sequence_padding(batch_segment_ids)
#         y_pred = model.predict([batch_token_ids, batch_segment_ids])
#         y_pred = y_pred[:, mask_idx, [neg_id, pos_id]]
#         # y_pred.argmax(axis=1)
#         scores = [x[1] for x in y_pred]
#         max_score = max(scores)
#         max_idx = np.argmax(scores)
#         pri = ZXSOP_knowledge[max_idx]

#         if max_score>thredshold:
#             sop_predict[curr_context_id].append(pri)

#             if curr_context_id in sop_true_dict:
#                 if dialogue_input_data[-1] in list(sop_true_dict[curr_context_id].keys()):
#                     if curr_context_id not in context_key_sop:
#                         context_key_sop[curr_context_id]={dialogue_input_data[-1]:pri}
#                     else:
#                         context_key_sop[curr_context_id].update({dialogue_input_data[-1]:pri})

#             if len(dialogue_input_data)<topN:
#                 context_top_sop[curr_context_id].append(pri)
    
#     return sop_predict,context_top_sop,context_key_sop


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
    for context_id in sop_true:
        res_true=list(set(sop_true[context_id]))
        if len(res_true)>1:
            res_true=[raw for raw in res_true if raw!=None]
        if (context_id in sop_predict and sop_predict[context_id]!=[]):
            res_predict = list(set(sop_predict[context_id]))
        else:
            res_predict = [None]

        if res_predict !=[None] and res_true == res_predict:
            for sop_t in res_true:
                context_true.extend([sop_list.index(sop_t)])
                context_pred.extend([sop_list.index(sop_t)])
        else:
            sop_max=res_predict if len(res_predict) >= len(res_true) else res_true
            for sop_index in range(len(sop_max)):
                if sop_index < len(res_true):
                    if res_true[sop_index] in sop_list:
                        context_true.extend([sop_list.index(res_true[sop_index])])
                    else:
                        context_true.extend([sop_list.index("None")])
                else:
                    context_true.extend([sop_list.index("None")])
                
                if sop_index < len(res_predict):
                    if res_predict[sop_index] in sop_list:
                        context_pred.extend([sop_list.index(res_predict[sop_index])])
                    else:
                        context_pred.extend([sop_list.index("None")])
                else:
                    context_pred.extend([sop_list.index("None")])
    
    assert len(context_true) == len(context_pred),"len(context_true):%s ,len(context_pred):%s "%(len(context_true),len(context_pred))
    f1=classification_report(context_true, context_pred,target_names=sop_list ,digits=4)
    return f1


def context_nlu_key_metric(sop_true_dict,context_key_sop,sop_true):
    key_true_list=[]
    key_pred_list=[]
    sop_list=[]
    for v in sop_true.values():
        sop_list.extend(v)
    sop_list=list(set([str(raw) for raw in sop_list]))
    # print(sop_list)
    for context_id in sop_true_dict:
        if context_id in context_key_sop:
            for sop_text in sop_true_dict[context_id]:
                key_true_list.append(sop_list.index(str(sop_true_dict[context_id][sop_text])))
                if sop_text in context_key_sop[context_id]:
                    # key_pred_list.append(sop_list.index(str(context_key_sop[context_id][sop_text])))
                    try:
                        key_pred_list.append(sop_list.index(str(context_key_sop[context_id][sop_text])))
                    except Exception as e:
                        key_pred_list.append(sop_list.index("None"))
                        print("分类思路问题：", e)
                else:
                    key_pred_list.append(sop_list.index("None"))
        else:
            for sop_text in sop_true_dict[context_id]:
                key_true_list.append(sop_list.index(str(sop_true_dict[context_id][sop_text])))
                key_pred_list.append(sop_list.index("None"))
    
    f1=classification_report(key_true_list, key_pred_list,target_names=sop_list ,digits=4)
    return f1


if __name__ == "__main__":

    sop_true,sop_nlu_predict,sop_true_dict = read_true_sop("data/推荐SOP数据804-811-手动进入.xlsx")
    nlu_bad_context_id = nlu_metric(sop_nlu_predict, sop_true)
    # nlu_acc: 0.5099255583126551

    model,tokenizer = None,None
    sop_predict,sop_top_predict,context_key_sop = dialogue_predict("data/sop_test_0825_data.xlsx", model, tokenizer, sop_true_dict)
    bad_context_id = context_nlu_metric(sop_predict, sop_true)
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
    f1=context_nlu_key_metric(sop_true_dict,context_key_sop,sop_true)
    print("dialogue key sentence:")
    print(f1)


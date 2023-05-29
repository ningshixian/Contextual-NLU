
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "0, 1"
import pickle
import re  
import json
import random
import pandas as pd
import numpy as np
import codecs
import scipy.stats
import tensorflow as tf
from bert4keras.backend import keras, K, search_layer, set_gelu
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.layers.normalization import BatchNormalization
from keras.constraints import unit_norm
from tqdm import tqdm
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold

from utils.DataPersistence import *
from optimizers import get_opt
# from margin_softmax import sparse_amsoftmax_loss


"""命令
nohup python train_pet.py > logs/train_pet_log.txt 2>&1 &
https://github.com/bojone/Pattern-Exploiting-Training/blob/master/sentiment.py
https://github.com/bojone/oppo-text-match/blob/main/baseline.py
"""

# sets random seed
seed = 123
np.random.seed(seed)
# set GPU memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 按需求增长
sess = tf.Session(config=config)
keras.backend.tensorflow_backend.set_session(sess)
# 切换gelu版本
set_gelu('tanh')  

# 基本参数
max_q_len = 128  # 32
batch_size = 16  # 16
epochs = 5  # 100
num_context = 5     # 上下文窗口
scale, margin = 30, 0.35  # amsoftmax参数 0.15 0.25 0.35
# scale, margin = 20, 0.25  # amsoftmax参数 0.15 0.25 0.35

# specify the batch size and number of epochs
LR = 2e-5  # 用足够小的学习率[3e-4, 5e-5, 2e-5] (adam默认学习率是0.001)
SGD_LR = 0.001
warmup_proportion = 0.1  # 学习率预热比例
weight_decay = 0.01  # 权重衰减系数，类似模型正则项策略，避免模型过拟合
DROPOUT_RATE = 0.5  # 0.2
optimizer = "adamwlr"

# 模型路径
# pm_root = "../corpus/chinese_L-12_H-768_A-12"    # BERT_base
# pm_root = "../corpus/chinese_wwm_ext_L-12_H-768_A-12",    # 如果选择哈工大中文模型，则设置LR=5e-5
pm_root = "../corpus/chinese_roberta_wwm_ext_L-12_H-768_A-12"    # roBERTa_base
# pm_root = "../corpus/longforBERT_v3.1",
# pm_root = "../corpus/mixed_corpus_bert_base_model"  # UER bert
# pm_root = "../corpus/chinese_simbert_L-12_H-768_A-12",    # SimBERT
# pm_root = "../corpus/chinese_roformer-sim-char-ft_L-12_H-768_A-12"    # SimBERT v2

config_path = pm_root + "/bert_config.json"
checkpoint_path = pm_root + "/bert_model.ckpt"
dict_path = pm_root + "/vocab.txt"

knowledge_path = "data/knowledge_20220901.xlsx"
# knowledge_path = "data/knowledge_20220826.xlsx"
training_data_path = "data/train_datasets_true_prod.json"
train_model_path = 'model/pet.weights'


def pre_tokenize(text):
    """单独识别出[xxx]的片段, 预分词
    """
    tokens, start = [], 0
    for r in re.finditer("\[[^\[]+\]", text):
        tokens.append(text[start : r.start()])
        tokens.append(text[r.start() : r.end()])
        start = r.end()
    if text[start:]:
        tokens.append(text[start:])
    return tokens


# 建立分词器，添加特殊占位符
# 如何正确地在vocab里增加一些token？ #408
# # 怎么把special token人为的插入？ #403
token_dict = load_vocab(dict_path=dict_path)
pure_tokenizer = Tokenizer(token_dict.copy(), do_lower_case=True)  # 全小写
user_dict = ["[pic]", "[http]", "[json]", "[phone]", "[subphone]"]  # special token只能是小写的
# user_dict = ["[pic]", "[http]", "[json]", "[time]", "[phone]", "[subphone]", "[alnum]", "[know]"]  # special token只能是小写的
for special_token in user_dict:
    if special_token not in token_dict:
        token_dict[special_token] = len(token_dict)
compound_tokens = [pure_tokenizer.encode(s)[0][1:-1] for s in user_dict]
tokenizer = Tokenizer(token_dict, do_lower_case=True, pre_tokenize=pre_tokenize)


# # 测试分词器-特殊字符
# print(tokenizer.encode('你好', '你也好'))
# print(tokenizer.tokenize("句子一[pic]句子二"))
# print(tokenizer.tokenize("句子一[PIC]句子二"))
# print(tokenizer.tokenize("北京是[unused1]中国的首都"))
# print(tokenizer.tokenize("[PAD][JSON][PIC]/你好呀 请问中石化充卡多久可以到账呢"))
# q1, q2 = "句子一[SEP]句子二".split("[SEP]")[-2:]
# # 测试分词器
# a_ids = tokenizer.encode(q1)[0]
# b_ids = tokenizer.encode(q2)[0][1:]
# token_ids = a_ids + b_ids
# segment_ids = [0] * len(a_ids)
# segment_ids += [1] * len(b_ids)
# print((token_ids, segment_ids))
# exit()


def trans_to_df(fcc_data):
    fcc_data_2 = {"context_id":[], "data":[], "label":[]}
    for k,v in fcc_data.items():
        fcc_data_2["context_id"].append(k)
        # data = list(map(lambda x:x[0], v["data"]))  # 取字符串，丢 calltype
        fcc_data_2["data"].append(v["data"])
        fcc_data_2["label"].append(v["label"])
    df = pd.DataFrame(fcc_data_2)
    return df


# 加载数据集
data = load_from_json(training_data_path)
df = trans_to_df(data)
print(df.shape)  # (3255, 3)
print(df.columns.values.tolist())
print(df[:2])


def clean(x):
    """预处理：去除文本的噪声信息"""
    import string
    from zhon.hanzi import punctuation
    # text, calltype = x
    # x = re.sub('"', "", str(x))
    # x = re.sub("\s", "", x)  # \s匹配任何空白字符，包括空格、制表符、换页符等
    # x = re.sub('[{}]+$'.format(string.punctuation+punctuation+" "), '', x)  # 干掉字符串结尾的标点（会干扰特殊占位符 "[PIC]"→"[PIC"）
    return x.strip()


def clean_sim(x):
    """预处理：切分相似问"""
    x = re.sub(r"(\t\n|\n)", "", x)
    x = x.strip().strip("###").replace("######", "###")
    return x.split("###")


def get_hard_neg_from_qa_pair(qa_pair: list):
    """负例获取
    以「非」当前回复内容作为负例
    注意：容易产生令人混淆的负例！
    """
    res = []
    index = [x for x in range(len(qa_pair))]
    random.shuffle(index)
    for i in range(0, len(qa_pair)):
        if i != index[i] and qa_pair[i][1] != qa_pair[index[i]][1]:
            res.append((qa_pair[i][0], qa_pair[index[i]][1], 0))
    return res


def build_training_data(data):
    D = []
    kid2label = {}
    kid2primary = {}
    pri2sim = {}
    
    # 线上知识库问题
    know = readExcel(knowledge_path, tolist=False)
    for index, row in tqdm(know.iterrows()):
        # if index==3000:
        #     break
        if row["base_code"] in ["XHTXBASE"]:    # 语料质量不佳
            continue
        if row["base_code"] in ["XIANLIAOBASE","CSZSK","LZYXCS","LZSCCSBASE","MSFTXIANLIAOBASE"]:  # 丢弃闲聊base语料
            continue
        pri = clean(row["primary_question"])
        sims = clean_sim(row["similar_question"])
        sims = list(set(filter(lambda x: x and "#" not in x and len(x)>1, sims)))
        kid = row["knowledge_id"]
        kid2label.setdefault(str(kid), len(kid2label))
        kid2primary.setdefault(str(kid), pri)
        pri2sim.setdefault(pri, [])
        for s in sims:
            # context = [['', "in"]]*(num_context-1) + [[clean(s), "in"]]
            label = kid2label[str(kid)]
            # D.append((context, label))
            D.append((s, pri, 1))
            pri2sim[pri].append(s)

    # 坐席日志数据挖掘
    for index, row in  tqdm(data.iterrows()): 
        context_id = row["context_id"]
        # 数据清洗
        func2 = lambda x: (clean(x), "in") if isinstance(x,str) else (clean(x[0]), x[1])
        row["data"] = list(map(func2, row["data"]))
        # 选取触发 SOP 的句子及其上文，构造训练集
        pre_idx = 0
        for item in row["label"]: 
            for k,v in item.items():
                intent,intent_id,sop_code = v
                if int(k)+1-pre_idx >= num_context:
                    pre_idx = int(k)+1-num_context
                elif int(k)+1 == pre_idx:
                    D.pop(-1)
                # context = row["data"][pre_idx:int(k)+1]
                # # 补齐上下文
                # if len(context)<num_context:
                #     context = [['', "in"]]*(num_context-len(context)) + context
                kid2label.setdefault(str(intent_id), len(kid2label))
                # kid2primary.setdefault(str(intent_id), intent)
                label = kid2label[str(intent_id)]
                # D.append((context, label))
                # pre_idx = int(k)+1
                kid2primary.setdefault(str(intent_id), intent)
                pri2sim.setdefault(pri, [])
                context = list(map(lambda x:x[0], row["data"][pre_idx:int(k)+1]))
                context = "[SEP]".join(context)  # 用句号将连续回复拼接起来 '[SEP]' '。'
                D.append((context, intent, 1))
                pre_idx = int(k)+1
    
    return D, kid2label, kid2primary, pri2sim


training_pkl_path = "model/training_data.pkl"
kid2label, label2kid = {}, {}  # kid转换成递增的id格式
D, kid2label, kid2primary, pri2sim = build_training_data(df)
D += get_hard_neg_from_qa_pair(D)
random.shuffle(D)   # 打乱数据集
label2kid = {v:k for k,v in kid2label.items()}
num_classes = len(kid2label)    # 类别数
print("\n数据加载完成, 共{}条".format(len(D)))  # 23334条
print(D[:5])

# 保存
joblib.dump((num_context, num_classes, D, kid2primary, label2kid), training_pkl_path)
with open("model/num_classes.txt", "w", encoding="utf-8") as f:  # 分类数cls_num的值保存到文件
    f.write(str(num_classes))
with open("model/D.txt", "w") as f:
    for item in D:
        f.write(str(item))
        f.write("\n")


# 对应的任务描述
prefix = u'很相似。'
mask_idx = 1
pos_id = tokenizer.token_to_id(u'很')
neg_id = tokenizer.token_to_id(u'不')


def random_masking(token_ids):
    """对输入进行随机mask
    """
    rands = np.random.random(len(token_ids))
    source, target = [], []
    for r, t in zip(rands, token_ids):
        if r < 0.15 * 0.8:
            source.append(tokenizer._token_mask_id)
            target.append(t)
        elif r < 0.15 * 0.9:
            source.append(t)
            target.append(t)
        elif r < 0.15:
            source.append(np.random.choice(tokenizer._vocab_size - 1) + 1)
            target.append(t)
        else:
            source.append(t)
            target.append(0)
    return source, target


# def sample_convert(text1, text2, label, random=False):
#     """转换为MLM格式
#     """
#     text1_ids = [tokens.get(t, 1) for t in text1]
#     text2_ids = [tokens.get(t, 1) for t in text2]
#     if random:
#         if np.random.random() < 0.5:
#             text1_ids, text2_ids = text2_ids, text1_ids
#         text1_ids, out1_ids = random_mask(text1_ids)
#         text2_ids, out2_ids = random_mask(text2_ids)
#     else:
#         out1_ids = [0] * len(text1_ids)
#         out2_ids = [0] * len(text2_ids)
#     token_ids = [2] + text1_ids + [3] + text2_ids + [3]
#     segment_ids = [0] * len(token_ids)
#     output_ids = [label + 5] + out1_ids + [0] + out2_ids + [0]
#     return token_ids, segment_ids, output_ids


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=True):
        batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []
        for is_end, (text1, text2, label) in self.sample(random):
            if label != 2:
                text = prefix + text1 + "[SEP]" + text2
            token_ids, segment_ids = tokenizer.encode(text, maxlen=max_q_len)
            if random:
                source_ids, target_ids = random_masking(token_ids)
            else:
                source_ids, target_ids = token_ids[:], token_ids[:]
            if label == 0:
                source_ids[mask_idx] = tokenizer._token_mask_id
                target_ids[mask_idx] = neg_id
            elif label == 1:
                source_ids[mask_idx] = tokenizer._token_mask_id
                target_ids[mask_idx] = pos_id
            batch_token_ids.append(source_ids)
            batch_segment_ids.append(segment_ids)
            batch_output_ids.append(target_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_output_ids = sequence_padding(batch_output_ids)
                yield [
                    batch_token_ids, batch_segment_ids, batch_output_ids
                ], None
                batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []


# 转换数据集
split_idx = int(len(D) * 0.8)
train_generator = data_generator(D[:split_idx], batch_size)
valid_generator = data_generator(D[split_idx:], batch_size)


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
        accuracy = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        accuracy = K.sum(accuracy * y_mask) / K.sum(y_mask)
        self.add_metric(accuracy, name='accuracy')
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


# 加载预训练模型
bert = build_transformer_model(
    config_path,
    checkpoint_path,
    with_mlm=True,
    compound_tokens=compound_tokens,  # 增加词，用字平均来初始化
)

# 训练用模型
y_in = keras.layers.Input(shape=(None,))
outputs = CrossEntropy(1)([y_in, bert.output])

train_model = keras.models.Model(bert.inputs + [y_in], outputs)
train_model.compile(optimizer=get_opt(optimizer, LR))
# train_model.compile(loss=masked_crossentropy, optimizer=Adam(1e-5))


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        # bert.save_weights('mlm_model.weights')
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            bert.save_weights(train_model_path)
        # test_acc = evaluate(test_generator)
        # print(
        #     u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
        #     (val_acc, self.best_val_acc, test_acc)
        # )


def evaluate(data):
    total, right = 0., 0.
    for x_true, _ in data:
        x_true, y_true = x_true[:2], x_true[2]
        y_pred = bert.predict(x_true)
        y_pred = y_pred[:, mask_idx, [neg_id, pos_id]].argmax(axis=1)
        y_true = (y_true[:, mask_idx] == pos_id).astype(int)
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


if __name__ == "__main__":

    # evaluator = Evaluator()

    # train_model.fit_generator(
    #     train_generator.forfit(),
    #     steps_per_epoch=len(train_generator),
    #     epochs=epochs,
    #     callbacks=[evaluator]
    # )

    bert.load_weights(train_model_path)
    text = prefix + "珑珠积分补录" + "[SEP]" + "今天天气不错"
    token_ids, segment_ids = tokenizer.encode(text, maxlen=max_q_len)
    source_ids, target_ids = token_ids[:], token_ids[:]
    source_ids[mask_idx] = tokenizer._token_mask_id
    # target_ids[mask_idx] = pos_id
    y_pred = bert.predict([[source_ids], [segment_ids]])
    y_pred = y_pred[:, mask_idx, [neg_id, pos_id]]
    print(y_pred)
    print(y_pred.argmax(axis=1))

    from evaluation import *
    sop_true,sop_nlu_predict,sop_text_name = read_true_sop("data/推荐SOP数据804-811-手动进入.xlsx")
    nlu_bad_context_id = nlu_metric(sop_nlu_predict, sop_true)
    sop_predict,sop_top_predict,context_key_sop = dialogue_predict_3("data/sop_test_0825_data.xlsx", bert, tokenizer, sop_text_name)
    bad_context_id = context_nlu_metric(sop_predict, sop_true)
    bad_context_id_top=context_nlu_metric(sop_top_predict,sop_true)
    # tmp = {}
    # for k,v in context_key_sop.items():
    #     tmp.setdefault(k, [])
    #     tmp[k].extend(v.values())
    # bad_context_id_top=context_nlu_metric(tmp,sop_true)

    f1=context_nlu_prf(sop_predict,sop_true)
    print("dialogue:")
    print(f1)
    f1_topN=context_nlu_prf(sop_top_predict,sop_true)
    print("dialogue Top5:")
    print(f1_topN)
    # f1=context_nlu_key_metric(sop_text_name,context_key_sop,sop_true)
    # print("dialogue key sentence:")
    # print(f1)

else:

    data, df, D, train_generator = None, None, None, None
    bert.load_weights(train_model_path)



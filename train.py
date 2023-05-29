
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "0, 1"
import pickle
import re  
import json
import pandas as pd
import numpy as np
import codecs
import random
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
from tqdm import tqdm
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold

from utils.DataPersistence import *
from optimizers import get_opt


"""命令
nohup python train.py > logs/train_log.txt 2>&1 &
https://github.com/bojone/bert4keras/blob/master/examples/task_sentiment_albert.py
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
max_q_len = 32
batch_size = 16
epochs = 5
num_context = 5     # 上下文窗口
scale, margin = 30, 0.35  # amsoftmax参数 0.15 0.25 0.35

# specify the batch size and number of epochs
LR = 2e-5  # 用足够小的学习率[3e-4, 5e-5, 2e-5] (adam默认学习率是0.001)
SGD_LR = 0.001
warmup_proportion = 0.1  # 学习率预热比例
weight_decay = 0.01  # 权重衰减系数，类似模型正则项策略，避免模型过拟合
DROPOUT_RATE = 0.5  # 0.2
optimizer = "adam"

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
training_data_path = "data/train_datasets_true_prod.json"
train_model_path = "model/best_model.weights"


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


def build_training_data(data):
    D = []
    kid2label = {}
    kid2primary = {}
    
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
        sims = clean_sim(row["similar_question"])   #[:10]
        sims = list(set(filter(lambda x: x and "#" not in x and len(x)>1, sims)))
        kid = row["knowledge_id"]
        kid2label.setdefault(str(kid), len(kid2label))
        kid2primary.setdefault(str(kid), pri)
        for s in sims:
            context = [['', "in"]]*(num_context-1) + [[clean(s), "in"]]
            label = kid2label[str(kid)]
            D.append((context, label))

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
                context = row["data"][pre_idx:int(k)+1]
                # 补齐上下文
                if len(context)<num_context:
                    context = [['', "in"]]*(num_context-len(context)) + context
                kid2label.setdefault(str(intent_id), len(kid2label))
                kid2primary.setdefault(str(intent_id), intent)
                label = kid2label[str(intent_id)]
                D.append((context, label))
                pre_idx = int(k)+1
    
    return D, kid2label, kid2primary


training_pkl_path = "model/training_data.pkl"
kid2label, label2kid = {}, {}  # kid转换成递增的id格式
D, kid2label, kid2primary = build_training_data(df)
random.shuffle(D)   # 打乱数据集
label2kid = {v:k for k,v in kid2label.items()}
num_classes = len(kid2label)    # 类别数
print("\n数据加载完成, 共{}条".format(len(D)))  #  35011条
print(D[:5])

# 保存
joblib.dump((num_context, num_classes, D, kid2primary, label2kid), training_pkl_path)
with open("model/num_classes.txt", "w", encoding="utf-8") as f:  # 分类数cls_num的值保存到文件
    f.write(str(num_classes))
with open("model/D.txt", "w") as f:
    for item in D:
        f.write(str(item))
        f.write("\n")


class data_generator(DataGenerator):
    """数据生成器
    单条样本：[CLS] Q1 [SEP] Q2 [SEP] Q3 [SEP]
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (q, label) in self.sample(random):
            token_ids, _ = tokenizer.encode(q[0][0], maxlen=max_q_len)
            segment_ids = [0] * len(token_ids) if q[0][1]=="in" else [1] * len(token_ids)
            for item in q[1:]:
                sen, calltype = item
                a,b = tokenizer.encode(sen, maxlen=max_q_len)
                token_ids += a[1:]
                segment_ids += [0] * (len(a)-1) if calltype=="in" else [1] * (len(a)-1)
            # token_ids, _ = tokenizer.encode(q[0], q[1], maxlen=2 * max_q_len)
            # token_ids += tokenizer.encode(q[2], maxlen=max_q_len)[0][1:]
            # segment_ids = [0] * len(token_ids)

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 转换数据集
split_idx = int(len(D) * 0.8)
train_generator = data_generator(D[:split_idx], batch_size)
valid_generator = data_generator(D[split_idx:], batch_size)


def adversarial_training(model, embedding_name, epsilon=1):
    """给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

    # 封装为函数
    inputs = (
        model._feed_inputs + model._feed_targets + model._feed_sample_weights
    )  # 所有输入层
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )  # 封装为函数

    def train_function(inputs):  # 重新定义训练函数
        grads = embedding_gradients(inputs)[0]  # Embedding梯度
        delta = epsilon * grads / (np.sqrt((grads**2).sum()) + 1e-8)  # 计算扰动
        K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        return outputs

    model.train_function = train_function  # 覆盖原训练函数


def sparse_categorical_crossentropy(y_true, y_pred):
    """自定义稀疏交叉熵
    这主要是因为keras自带的sparse_categorical_crossentropy不支持求二阶梯度。
    """
    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
    y_true = K.cast(y_true, 'int32')
    y_true = K.one_hot(y_true, K.shape(y_pred)[-1])
    return K.categorical_crossentropy(y_true, y_pred)


def loss_with_gradient_penalty(y_true, y_pred, epsilon=1):
    """带梯度惩罚的loss
    https://github.com/bojone/bert4keras/blob/master/examples/task_iflytek_gradient_penalty.py
    """
    loss = K.mean(sparse_categorical_crossentropy(y_true, y_pred))
    embeddings = search_layer(y_pred, 'Embedding-Token').embeddings
    gp = K.sum(K.gradients(loss, [embeddings])[0].values**2)
    return loss + 0.5 * epsilon * gp


# 加载预训练模型
bert = build_transformer_model(
    config_path,
    checkpoint_path,
    # model="bert",  # roformer
    # with_pool=True,
    # application='encoder',
    compound_tokens=compound_tokens,  # 增加词，用字平均来初始化
    return_keras_model=False,
)

output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
output = Dropout(DROPOUT_RATE, name="drop")(output)   #防止过拟合
output = Dense(
    units=num_classes,
    activation='softmax',
    kernel_initializer=bert.initializer
)(output)

model = keras.models.Model(bert.model.input, output)
# model.summary()

model.compile(
    loss='sparse_categorical_crossentropy',
    # loss=loss_with_gradient_penalty,
    optimizer=get_opt(optimizer, LR),  # 用足够小的学习率
    metrics=['accuracy'],   # sparse_categorical_accuracy
)

# # 启用对抗训练只需要一行代码
# # https://github.com/bojone/bert4keras/tree/master/examples/task_iflytek_adversarial_training.py
# adversarial_training(model, 'Embedding-Token', 0.5)


# class Evaluator(keras.callbacks.Callback):
#     """评估与保存
#     """
#     def __init__(self):
#         self.lowest = 1

#     def on_epoch_end(self, epoch, logs=None):
#         # 保存最优
#         if logs["loss"] <= self.lowest:
#             print("保存当前模型...")
#             self.lowest = logs["loss"]
#             model.save_weights(train_model_path)
#         # # 演示效果
#         # just_show()


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    """保存验证集acc最好的模型
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights(train_model_path)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f\n' %
            (val_acc, self.best_val_acc)
        )


if __name__ == "__main__":

    # evaluator = Evaluator()

    # model.fit(
    #     train_generator.forfit(),
    #     steps_per_epoch=len(train_generator),
    #     epochs=epochs,
    #     callbacks=[evaluator]
    # )

    from evaluation import *
    model.load_weights(train_model_path)
    # sop_true,sop_nlu_predict,sop_text_name = read_true_sop("data/知识推荐数据-0902-516条-未审核-知识搜索.xlsx")
    # sop_predict,sop_top_predict,context_key_sop = dialogue_predict("data/knowledge_recommend_test_0902_data.xlsx", model, tokenizer, sop_text_name)
    sop_true,sop_nlu_predict,sop_text_name = read_true_sop("data/推荐SOP数据804-811-手动进入.xlsx")
    nlu_bad_context_id = nlu_metric(sop_nlu_predict, sop_true)
    sop_predict,sop_top_predict,context_key_sop = dialogue_predict("data/sop_test_0825_data.xlsx", model, tokenizer, sop_text_name)
    bad_context_id = context_nlu_metric(sop_predict, sop_true)
    bad_context_id_top=context_nlu_metric(sop_top_predict,sop_true)
    # tmp = {}
    # for k,v in context_key_sop.items():
    #     tmp.setdefault(k, [])
    #     tmp[k].extend(v.values())
    # bad_context_id_top=context_nlu_metric(tmp,sop_true)

    # for raw in sop_predict:
    #     if set(sop_predict[raw])!=set(sop_top_predict[raw]):
    #         print(raw,set(sop_predict[raw]),set(sop_top_predict[raw]))

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
    model.load_weights(train_model_path)



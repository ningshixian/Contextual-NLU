
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "0, 1"
import pickle
import re  
import json
import random
import pandas as pd
import numpy as np
import codecs
import scipy.stats
from scipy.optimize import minimize
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
from margin_softmax import sparse_amsoftmax_loss


"""命令
python train_cosent.py RoBERTa cls LXH 0.3
nohup python train_cosent.py RoBERTa cls LXH 0.3 > logs/train_cosent_log.txt 2>&1 &
https://github.com/bojone/CoSENT/blob/main/cosent.py
"""

# sets random seed
seed = 123
random.seed(seed)
np.random.seed(seed)
# set cuda device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
epochs = 15  # 100
num_context = 5     # 上下文窗口
scale, margin = 30, 0.35  # amsoftmax参数 20, 0.25
t = 0.05  # 温度系数τ

# specify the batch size and number of epochs
LR = 2e-5  # 用足够小的学习率[3e-4, 5e-5, 2e-5] (adam默认学习率是0.001)
SGD_LR = 0.001
warmup_proportion = 0.1  # 学习率预热比例
weight_decay = 0.01  # 权重衰减系数，类似模型正则项策略，避免模型过拟合
optimizer = "adamwlr"

# 训练集路径
# knowledge_path = "data/knowledge_20220901.xlsx"
# knowledge_path = "data/knowledge_20220826.xlsx"
knowledge_lzzxfz_path = "data/knowledge_LZZXFZ.xlsx"
knowledge_robot_path = "data/knowledge_robot.xlsx"
training_data_path = "data/train_datasets_true_prod.json"
train_model_path = 'model/cosent.weights'


if len(sys.argv[1:]) == 4:
    model_type, pooling, task_name, DROPOUT_RATE = sys.argv[1:]
else:
    model_type, pooling, task_name, DROPOUT_RATE = "BERT", "cls", "LXH", 0.3
assert model_type in [
    "BERT",
    "RoBERTa",
    "NEZHA",
    "WoBERT",
    "RoFormer",
    "BERT-large",
    "RoBERTa-large",
    "NEZHA-large",
    "SimBERT",
    "SimBERT-tiny",
    "SimBERT-small",
    "SimBERT-v2",
    "UER",
    "LongforBERT",
]
assert pooling in ["first-last-avg", "last-avg", "cls", "pooler"]
assert task_name in ["ATEC", "BQ", "LCQMC", "PAWSX", "STS-B", "LXH"]
DROPOUT_RATE = float(DROPOUT_RATE)

# bert配置
model_name = {
    "BERT": "chinese_L-12_H-768_A-12",    # BERT_base
    "RoBERTa": "chinese_roberta_wwm_ext_L-12_H-768_A-12",    # roBERTa_base
    "WoBERT": "chinese_wobert_plus_L-12_H-768_A-12",
    "NEZHA": "nezha_base_wwm",
    "RoFormer": "chinese_roformer_L-12_H-768_A-12",
    "BERT-large": "uer/mixed_corpus_bert_large_model",
    "RoBERTa-large": "chinese_roberta_wwm_large_ext_L-24_H-1024_A-16",
    "NEZHA-large": "nezha_large_wwm",
    "SimBERT": "chinese_simbert_L-12_H-768_A-12",
    "SimBERT-tiny": "chinese_simbert_L-4_H-312_A-12",
    "SimBERT-small": "chinese_simbert_L-6_H-384_A-12",
    "SimBERT-v2": "chinese_roformer-sim-char-ft_L-12_H-768_A-12",    # SimBERT v2
    "UER": "mixed_corpus_bert_base_model",
    "LongforBERT": "longforBERT_v4.1",
}[model_type]

config_path = "../corpus/%s/bert_config.json" % model_name
if model_type == "NEZHA":
    checkpoint_path = "../corpus/%s/model.ckpt-691689" % model_name
elif model_type == "NEZHA-large":
    checkpoint_path = "../corpus/%s/model.ckpt-346400" % model_name
else:
    checkpoint_path = "../corpus/%s/bert_model.ckpt" % model_name
dict_path = "../corpus/%s/vocab.txt" % model_name


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
if model_type in ["WoBERT", "RoFormer"]:
    tokenizer = Tokenizer(token_dict, do_lower_case=True, pre_tokenize=lambda s: jieba.lcut(s, HMM=False))
else:
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
            res.append((qa_pair[i][0], qa_pair[index[i]][1], 0, (qa_pair[i][3][0], qa_pair[index[i]][3][0])))
    return res


def build_training_data(data):
    D = []
    kid2label = {}
    kid2primary = {}
    pri2sim, pri2base = {}, {}
    
    # 线上知识(主问题+相似问)
    for k_path in [knowledge_lzzxfz_path, knowledge_robot_path]:
        know = readExcel(k_path, tolist=False)
        for index, row in tqdm(know.iterrows()):
            if row["base_code"] in ["XHTXBASE"]:    # 语料质量不佳
                continue
            if row["base_code"] in ["XIANLIAOBASE","CSZSK","LZYXCS","LZSCCSBASE","MSFTXIANLIAOBASE"]:  # 丢弃闲聊base语料
                continue
            pri = clean(row["primary_question"])
            sims = clean_sim(row["similar_question"])   # [:20]
            sims = list(set(filter(lambda x: x and "#" not in x and len(x)>1, sims)))
            kid = row["knowledge_id"]
            kid2label.setdefault(str(kid), len(kid2label))
            kid2primary.setdefault(str(kid), pri)
            if k_path=="data/knowledge_LZZXFZ.xlsx":
                pri2base[pri] = row["base_code"]
                pri2sim.setdefault(pri, [])
                for s in sims:
                    label = kid2label[str(kid)]
                    D.append((s, pri, 1, (label, label)))
                    pri2sim[pri].append(s)
            else:
                for s in sims:
                    label = kid2label[str(kid)]
                    D.append((s, pri, 1, (label, label)))

    # 坐席日志对话数据
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
                pri2base[pri] = "ZXSOPBASE" if (sop_code and sop_code not in [None,"None","null"]) else "None"
                pri2sim.setdefault(pri, [])
                context = list(map(lambda x:x[0], row["data"][pre_idx:int(k)+1]))
                context = "[SEP]".join(context)  # 用句号将连续回复拼接起来 '[SEP]' '。'
                D.append((context, intent, 1, (label, label)))
                pre_idx = int(k)+1
    
    return D, kid2label, kid2primary, pri2sim, pri2base


training_pkl_path = "model/training_data.pkl"
kid2label, label2kid = {}, {}  # kid转换成递增的id格式
D, kid2label, kid2primary, pri2sim, pri2base = build_training_data(df)
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


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels, batch_classes = [], [], [], []
        for is_end, (text1, text2, label, class2) in self.sample(random):
            for i, text in enumerate([text1, text2]):
                token_ids, segment_ids = tokenizer.encode(text, maxlen=max_q_len)
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append([label])
                batch_classes.append([class2[i]])
            if len(batch_token_ids) == self.batch_size * 2 or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                batch_classes = sequence_padding(batch_classes)
                yield [batch_token_ids, batch_segment_ids], [batch_labels, batch_classes]
                batch_token_ids, batch_segment_ids, batch_labels, batch_classes = [], [], [], []


# 转换数据集
split_idx = int(len(D) * 0.8)
train_generator = data_generator(D[:split_idx], batch_size)
valid_generator = data_generator(D[split_idx:], batch_size)


class data_generator_4_simcse(DataGenerator):
    """训练语料生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text1, text2, label, class2) in self.sample(random):
            for i, text in enumerate([text1, text2]):
                token_ids, segment_ids = tokenizer.encode(text, maxlen=max_q_len)
                # 每个batch内，每一句话都重复一次
                for i in range(2):
                    batch_token_ids.append(token_ids)
                    batch_segment_ids.append(segment_ids)
                    batch_labels.append([label])

                if len(batch_token_ids) == self.batch_size * 2 or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_labels = sequence_padding(batch_labels)
                    yield [batch_token_ids, batch_segment_ids], batch_labels
                    batch_token_ids, batch_segment_ids, batch_labels = [], [], []


def custom_amsoftmax_loss(scale, margin):
    # 采用了闭包的方式，将参数传给 sparse_amsoftmax_loss，再调用 inner
    def inner(y_true, y_pred):
        return sparse_amsoftmax_loss(y_true,y_pred,scale, margin)
    return inner


def cosent_loss(y_true, y_pred):
    """排序交叉熵
    y_true：标签/打分，y_pred：句向量
    """
    # 1. 取出真实的标签
    y_true = y_true[::2, 0]
    # 2. 对输出的句子向量进行l2归一化   后面只需要对应为相乘  就可以得到cos值了
    y_pred = K.l2_normalize(y_pred, axis=1)
    # 3. 奇偶向量相乘, 相似度矩阵除以温度系数0.05(等于*20)
    y_pred = K.sum(y_pred[::2] * y_pred[1::2], axis=1) * 20
    # 4. 取出负例-正例的差值
    y_pred = y_pred[:, None] - y_pred[None, :]  # 这里是算出所有位置 两两之间余弦的差值
    # 矩阵中的第i行j列  表示的是第i个余弦值-第j个余弦值
    y_true = K.cast(y_true[:, None] < y_true[None, :], K.floatx())  # 取出负例-正例的差值
    y_pred = K.reshape(y_pred - (1 - y_true) * 1e12, [-1])
    y_pred = K.concatenate([[0], y_pred], axis=0)
    return K.logsumexp(y_pred)
    # return K.mean(K.logsumexp(y_pred))


def simcse_loss(y_true, y_pred):
    """用于SimCSE训练的loss
    y_true只是凑数的，并不起作用。因为真正的y_true是通过batch内数据计算得出的。
    y_pred就是batch内的每句话的embedding，通过bert编码得来
    """
    # 构造标签
    # idxs = [0,1,2,3,4,5]
    idxs = K.arange(0, K.shape(y_pred)[0])
    # 给idxs添加一个维度，idxs_1 = [[0,1,2,3,4,5]]
    idxs_1 = idxs[None, :]
    # 获取每句话的同义句id，即
    # 如果一个句子id为奇数，那么和它同义的句子的id就是它的上一句，如果一个句子id为偶数，那么和它同义的句子的id就是它的下一句
    # idxs_2 = [ [1], [0], [3], [2], [5], [4] ]
    idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
    # 生成计算loss时可用的标签
    # y_true = [[0,1,0,0,0,0],[1,0,0,0,0,0],[0,0,0,1,0,0],[0,0,1,0,0,0],[0,0,0,0,0,1],[0,0,0,0,1,0]]
    y_true = K.equal(idxs_1, idxs_2)
    y_true = K.cast(y_true, K.floatx())
    # 计算相似度
    # 首先对句向量各个维度做了一个L2正则，使其变得各项同性，避免下面计算相似度时，某一个维度影响力过大。
    y_pred = K.l2_normalize(y_pred, axis=1)
    # 其次，计算batch内每句话和其他句子的内积相似度(其实就是余弦相似度)
    similarities = K.dot(y_pred, K.transpose(y_pred))
    # 然后，将矩阵的对角线部分变为0，代表每句话和自身的相似性并不参与运算
    similarities = similarities - tf.eye(K.shape(y_pred)[0]) * 1e12
    # 温度系数τ=0.05
    similarities = similarities / t
    # from_logits=True的交叉熵自带softmax激活函数
    loss = K.categorical_crossentropy(y_true, similarities, from_logits=True)
    return K.mean(loss)


# 加载预训练模型
if model_type == "RoFormer":
    model_type = "roformer"
elif "NEZHA" in model_type:
    model_type = "nezha"
else:
    model_type = "bert"

bert = build_transformer_model(
    config_path,
    checkpoint_path,
    model=model_type,
    dropout_rate=DROPOUT_RATE,
    # with_pool=True,
    # application='encoder',
    compound_tokens=compound_tokens,  # 增加词，用字平均来初始化
    return_keras_model=False,
)
cls_embedding = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)

# r_output = Dense(
#     units=num_classes,
#     activation='softmax',
#     kernel_initializer=bert.initializer,
#     name = 'dense_layer'
# )(cls_embedding)

output = Lambda(lambda v: K.l2_normalize(v, 1))(cls_embedding)  # 特征归一化（l2正则）√
# output = Dropout(DROPOUT_RATE, name="dp1")(output)   #防止过拟合
output = Dense(
    units=num_classes,
    use_bias=False,  # no bias √
    kernel_constraint=unit_norm(),  # 权重归一化（单位范数（unit_form），限制权值大小为 1.0）√
    kernel_initializer=bert.initializer,
    name = 'dense_layer'
)(output)

encoder = keras.models.Model(bert.model.inputs, cls_embedding) # 最终的目的是要得到一个编码器
model = keras.models.Model(inputs=bert.model.inputs, outputs=[cls_embedding, output]) # 用多任务做训练
# model.summary()

# losses={'CLS-token': cosent_loss,'dense_layer': "sparse_categorical_crossentropy"}
losses={'CLS-token': cosent_loss,'dense_layer': custom_amsoftmax_loss(scale, margin)}
loss_weights={'CLS-token':1, 'dense_layer': 1}
model.compile(optimizer=get_opt(optimizer, LR), loss=losses, loss_weights=loss_weights)

# # 自定义的loss
# # 重点：把自定义的loss添加进层使其生效，同时加入metric方便在KERAS的进度条上实时追踪
# x1_in = Input(shape=(None,))
# x2_in = Input(shape=(None,))
# target = Input(shape=(None,), dtype="int32")
# model = keras.models.Model(inputs=[x1_in, x2_in], outputs=[embedding])
# am_loss = sparse_amsoftmax_loss(target, output, scale, margin)
# sim_loss = simcse_loss(target, emb)
# rdrop_loss = rdrop_loss(target, r_output)
# # 配合R-Drop的交叉熵损失
# ce_rdrop_loss = crossentropy_with_rdrop(target, r_output)
# # 配合R-Drop的amsoftmax损失
# am_rdrop_loss = K.mean(am_loss) + rdrop_loss
# # 配合SimCSE的amsoftmax损失
# am_simcse_loss = K.mean(am_loss) + sim_loss
# # All Three Loss 加权和
# am_simcse_rdrop_loss = K.mean(am_loss) + sim_loss + rdrop_loss
# # 联合训练 amsoftmax+RDrop 
# train_model.add_loss(am_rdrop_loss)
# train_model.add_metric(K.mean(am_loss), name="am_loss")
# train_model.add_metric(rdrop_loss, name="rdrop_loss")


def compute_corrcoef(x, y):
    """Spearman相关系数
    """
    return scipy.stats.spearmanr(x, y).correlation


def l2_normalize(vecs):
    """l2标准化
    """
    norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs / np.clip(norms, 1e-8, np.inf)


def optimal_threshold(y_true, y_pred):
    """最优阈值的自动搜索
    """
    loss = lambda t: -np.mean((y_true > 0.5) == (y_pred > np.tanh(t)))
    result = minimize(loss, 1, method='Powell')
    return np.tanh(result.x), -result.fun


class Evaluator(keras.callbacks.Callback):
    """保存验证集分数最好的模型
    """
    def __init__(self):
        self.best_accuracy = 0.
        self.best_spearman = 0.
        self.best_threshold = 0.

    def on_epoch_end(self, epoch, logs=None):
        spearman, accuracy, threshold = self.evaluate(valid_generator)
        if spearman > self.best_spearman:
            self.best_spearman = spearman
            self.best_threshold = threshold
            model.save_weights(train_model_path)
        print(
            u'spearman: %.5f, accuracy: %.5f, threshold: %.5f, best_spearman: %.5f\n'
            % (spearman, accuracy, threshold, self.best_spearman)
        )

    def evaluate(self, data, threshold=None):
        Y_true, Y_pred = [], []
        for x_true, y_true in data:
            Y_true.extend(y_true[0][::2, 0])
            x_vecs = encoder.predict(x_true)
            x_vecs = l2_normalize(x_vecs)
            y_pred = (x_vecs[::2] * x_vecs[1::2]).sum(1)
            Y_pred.extend(y_pred)
        Y_true, Y_pred = np.array(Y_true), np.array(Y_pred)
        spearman = compute_corrcoef(Y_true, Y_pred)
        if threshold is None:
            threshold, accuracy = optimal_threshold(Y_true, Y_pred)
        else:
            accuracy = np.mean((Y_true > 0.5) == (Y_pred > threshold))
        return spearman, accuracy, threshold


if __name__ == "__main__":

    # # 在领域数据上做SimCSE训练
    # encoder.compile(loss=simcse_loss, optimizer=Adam(1e-5))
    # train_generator_4_simcse = data_generator_4_simcse(D[:], batch_size)   # 数据格式[x1,x1,x2,x2,...]
    # encoder.fit(
    #     train_generator_4_simcse.forfit(), 
    #     steps_per_epoch=len(train_generator_4_simcse), 
    #     epochs=1
    # )

    # evaluator = Evaluator()

    # model.fit_generator(
    #     train_generator.forfit(),
    #     steps_per_epoch=len(train_generator),
    #     epochs=epochs,
    #     callbacks=[evaluator]
    # )
    # model.save_weights(train_model_path + ".last")

    # model.load_weights(train_model_path)
    # # encoder = Model(input=model.input, output=model.get_layer("CLS-token").output)
    # metrics = evaluator.evaluate(valid_generator, evaluator.best_threshold)
    # metrics = tuple(metrics[:2])
    # print(u'val spearman: %.5f, test accuracy: %.5f' % metrics)


    # # ===================================================================== # #

    # from evaluation import *
    # test_data_path = "data/算法上下文评测数据标注-审核数据-3026.xlsx"
    # sop_test_data = pd.read_excel(test_data_path)  # 读取到空字符串时非nan , keep_default_na=False
    # sop_test_data.fillna("", inplace=True)
    # context_id = sop_test_data["context_id"]
    # context_label = sop_test_data["context_label"]
    # p1 = sop_test_data["算法预测1"]
    # p2 = sop_test_data["算法预测2"]
    # p3 = sop_test_data["算法预测3"]
    # context_nlu_metric_2(p1, context_label)     # 句子维度
    # context_nlu_metric_2(p2, context_label)     # 句子维度
    # context_nlu_metric_2(p3, context_label)     # 句子维度
    # exit()

    # 仅使用 SOP 知识作为搜索范围，进行评测（弃用）
    # ZXSOP_knowledge=["珑珠积分补录","C2停车权益","订单未发货","权益变更","珑珠券","手机号修改流程","删除实名认证信息","账户无法登录","收不到验证码","C1签约未返珑珠","C1推荐未返珑珠","C3签约未返珑珠"
    #               ,"C3推荐未返珑珠","C5签约未返珑珠","C5推荐未返珑珠","C1/C4活动返珑珠","C4缴物业费返珑珠","话费充值","天猫超市卡充值","京东企业购（退换货）","珑珠优选（换货）","珑珠优选（退货）"]
    
    model.load_weights(train_model_path)
    # model.load_weights(train_model_path + ".last")
    print("\n坐席知识向量化...")
    pri2vec = {}
    for know, sims in pri2sim.items():
        pri2vec.setdefault(know, [])
        sims = pri2sim.get(know, []) + [know]
        batch_token_ids, batch_segment_ids = [], []
        for s in sims:
            token_ids, segment_ids = tokenizer.encode(s, maxlen=max_q_len)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
        batch_token_ids = sequence_padding(batch_token_ids)
        batch_segment_ids = sequence_padding(batch_segment_ids)
        vecs = encoder.predict([batch_token_ids, batch_segment_ids])
        pri2vec[know].extend(vecs)
    
    from evaluation import *
    test_data_path = "data/算法上下文评测数据标注-审核数据-3026.xlsx"
    sop_predict, sop_true, context_pridect, context_true = dialogue_predict_2(test_data_path, encoder, tokenizer, [], pri2vec, pri2base)
    context_nlu_metric(sop_predict,sop_true)    # context维度
    context_nlu_metric_2(context_pridect, context_true)     # 句子维度
    exit()

    # from evaluation import *
    # sop_true,sop_nlu_predict,sop_text_name = read_true_sop("data/推荐SOP数据804-811-手动进入.xlsx")
    # nlu_bad_context_id = nlu_metric(sop_nlu_predict, sop_true)
    # sop_predict,sop_top_predict,context_key_sop = dialogue_predict_2("data/sop_test_0825_data.xlsx", encoder, tokenizer, sop_text_name, pri2vec, pri2base)
    # bad_context_id = context_nlu_metric(sop_predict, sop_true)
    # bad_context_id_top=context_nlu_metric(sop_top_predict,sop_true)
    # # tmp = {}
    # # for k,v in context_key_sop.items():
    # #     tmp.setdefault(k, [])
    # #     tmp[k].extend(v.values())
    # # bad_context_id_top=context_nlu_metric(tmp,sop_true)

    # f1=context_nlu_prf(sop_predict,sop_true)
    # print("dialogue:")
    # print(f1)
    # f1_topN=context_nlu_prf(sop_top_predict,sop_true)
    # print("dialogue Top5:")
    # print(f1_topN)
    # # f1=context_nlu_key_metric(sop_text_name,context_key_sop,sop_true)
    # # print("dialogue key sentence:")
    # # print(f1)

else:

    data, df, D, train_generator = None, None, None, None
    model.load_weights(train_model_path)
    # model.load_weights(train_model_path + ".last")
    # encoder = Model(input=model.input, output=model.get_layer("CLS-token").output)


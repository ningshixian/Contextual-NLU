import re
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from utils.DataPersistence import *

knowledge_path = "data/knowledge_20220901.xlsx"
training_data_path = "data/datasets_true_prod.json"


def clean_sim(x):
    """预处理：切分相似问"""
    x = re.sub(r"(\t\n|\n)", "", x)
    x = x.strip().strip("###").replace("######", "###")
    return x.split("###")


# 加载数据集
data = load_from_json(training_data_path)
know = readExcel(knowledge_path, tolist=False)

sentences = []
for index, row in tqdm(know.iterrows()):
    if row["base_code"] in ["XHTXBASE"]:    # 语料质量不佳
        continue
    if row["base_code"] in ["XIANLIAOBASE","CSZSK","LZYXCS","LZSCCSBASE","MSFTXIANLIAOBASE"]:  # 丢弃闲聊base语料
        continue
    pri = row["primary_question"]
    sims = clean_sim(row["similar_question"])   #[:10]
    sims = list(set(filter(lambda x: x and "#" not in x and len(x)>1, sims)))
    sentences.extend([pri])
    sentences.extend(sims)
for k,v in data.items():
    sentences.extend(v["data"])

sentences = list(filter(lambda x:len(x)<50, sentences)) 
sentence_length = list(map(lambda x:len(x), sentences)) 

ax = sns.distplot(sentence_length)
# plt.show()#横轴是句子的长度 纵轴是概率密度
hist_fig = ax.get_figure()
hist_fig.savefig("fig.png", dpi = 400)


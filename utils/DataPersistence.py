import pickle as pkl
import json
import pandas as pd
import os

"""数据持久化存储和读取"""


def load_from_pkl(pkl_path):
    with open(pkl_path, "rb") as fin:
        obj = pkl.load(fin)
    return obj


def dump_to_pkl(obj, pkl_path):
    with open(pkl_path, "wb") as fout:
        pkl.dump(obj, fout, protocol=pkl.HIGHEST_PROTOCOL)


def load_from_json(json_path):
    data = None
    with open(json_path, "r", encoding="utf-8") as f:
        try:
            data = json.loads(f.read())
        except Exception as e:
            print(e)
            raise ValueError("%s is not a legal JSON file, please check your JSON format!" % json_path)
    return data


def dump_to_json(obj, json_path):
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj))


# 读取EXCEL文件
def readExcel(excel_file, sheet_name=0, tolist=True):
    obj = None
    if os.path.exists(excel_file):
        io = pd.io.excel.ExcelFile(os.path.join(excel_file))
        df = pd.read_excel(io, sheet_name)
        df.fillna("", inplace=True) # Fill NA/NaN values using the specified method.
        io.close()
        obj = df.values.tolist() if tolist else df
    return obj


# #  将数据写入EXCEL文件
# def writeExcel(file_path, datas):
#     import xlwt
#     f = xlwt.Workbook()
#     sheet1 = f.add_sheet(u"sheet1", cell_overwrite_ok=True)  # 创建sheet
#     for i in range(len(datas)):
#         data = datas[i]
#         for j in range(len(data)):
#             sheet1.write(i, j, str(data[j]))  # 将数据写入第 i 行，第 j 列
#     f.save(file_path)  # 保存文件


#  将数据写入EXCEL文件
def writeExcel(file_path, datas, header=[]):
    """
    file_path: 写入excel文件路径
    datas: 要写入的数据
    header 指定列名
    """
    writer = pd.ExcelWriter(file_path)
    if isinstance(datas, dict):
        # datas = {"col1": [1, 1], "col2": [2, 2]}
        df1 = pd.DataFrame(datas)
        df1.to_excel(writer, "Sheet1", index=False)
    elif isinstance(datas, list):
        # 二维数组型数据
        df = pd.DataFrame.from_records(list(datas))
        df.to_excel(
            writer, "Sheet1", index=False, header=header
        )
    writer.save()


def read_csv(file_path):
    csvFile = open(file_path)
    csv_reader = csv.reader(csvFile)
    c2_corpus = []
    for line in csv_reader:
        # 忽略第一行
        if csv_reader.line_num == 1:
            continue
        primary_question, similar_question, flow_code, key_word = line[1], line[2], line[4], line[5]
        if flow_code=="C2ZXKF":
            c2_corpus.append(primary_question)
            c2_corpus.extend([_ for _ in similar_question.split("###") if _])
    csvFile.close()
    return c2_corpus

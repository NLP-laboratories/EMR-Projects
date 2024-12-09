from glob import glob
import os
import random
import pandas as pd
from config import *


"""对input/origin中的ann文件预处理,取出实体名称及位置,索引位置与实体标注名称对应"""
  
def get_annotation(ann_path):
    with open(ann_path, encoding='utf-8') as file:
        anns = {}
        for line in file.readlines():
            arr = line.split('\t')[1].split()
            name = arr[0]
            start = int(arr[1])
            end = int(arr[-1])
            # 标注太长，可能有问题
            if end - start > 50:
                continue
            anns[start] = 'B-' + name
            for i in range(start + 1, end):
                anns[i] = 'I-' + name
        return anns




"""将text文件中的所有文字匹配对应的实体;形状DataFrame,第一列对应Word,第二列对应标签
将导出的文件存在annotation文件夹"""

def get_text(txt_path):
    with open(txt_path, encoding='utf-8') as file:
        return file.read()

def generate_annotation():
    for txt_path in glob(ORIGIN_DIR + '*.txt'):
        ann_path = txt_path[:-3] + 'ann'
        anns = get_annotation(ann_path)
        text = get_text(txt_path)
        # 建立文字和标注对应
        df = pd.DataFrame({'word': list(text), 'label': ['O'] * len(text)})
        df.loc[anns.keys(), 'label'] = list(anns.values())
        # 导出文件
        file_name = os.path.split(txt_path)[1]
        df.to_csv(ANNOTATION_DIR + file_name, header=None, index=None)



"""将annotation中的文件拆分为训练集和测试集"""

def split_sample(test_size=0.3):
    files = glob(ANNOTATION_DIR + '*.txt')
    random.seed(0)
    random.shuffle(files)
    n = int(len(files) * test_size)
    test_files = files[:n]
    train_files = files[n:]
    # 合并文件
    merge_file(train_files, TRAIN_SAMPLE_PATH)
    merge_file(test_files, TEST_SAMPLE_PATH)


def merge_file(files, target_path): # target_path：合并到的目标文件
    with open(target_path, 'a', encoding='utf-8') as file:
        for f in files:
            text = open(f, encoding='utf-8').read()
            file.write(text)


""" 根据训练集生成词表"""

def generate_vocab():
    df = pd.read_csv(TRAIN_SAMPLE_PATH, usecols=[0], names=['word'])  # 取出训练集中的文字列
    vocab_list = [WORD_PAD, WORD_UNK] + df['word'].value_counts().keys().tolist() # 按照词频构成vocab_list
    vocab_list = vocab_list[:VOCAB_SIZE]
    vocab_dict = {v: k for k, v in enumerate(vocab_list)} # 对vocab_list中的词匹配索引
    vocab = pd.DataFrame(list(vocab_dict.items())) # 字典转 DataFrame
    vocab.to_csv(VOCAB_PATH, header=None, index=None) # 将词表写到缓存文件


# 根据训练集生成标签表
def generate_label():
    df = pd.read_csv(TRAIN_SAMPLE_PATH, usecols=[1], names=['label'])
    label_list = df['label'].value_counts().keys().tolist()
    label_dict = {v: k for k, v in enumerate(label_list)}
    label = pd.DataFrame(list(label_dict.items()))
    label.to_csv(LABEL_PATH, header=None, index=None)


if __name__ == '__main__':


    # 位置索引与实体标注名称对应
    # anns = generate_annotation(r'input\origin\154_6.ann')
    # print(anns)
    # exit()

    # 将所有text文件中的文字对应一个实体并将导出的文件存在annotation文件夹中
    generate_annotation()


    # 拆分训练集和测试集
    split_sample()

#     # 生成词表
    generate_vocab()

#     # 生成标签表
    generate_label()
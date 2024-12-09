import torch
from torch.utils import data
from config import *
import pandas as pd
from seqeval.metrics import classification_report
from transformers import BertTokenizer
from transformers import logging

logging.set_verbosity_warning()



"""调用词表和标签表"""

def get_vocab():
    df = pd.read_csv(VOCAB_PATH, names=['word', 'id'])
    return list(df['word']), dict(df.values)


def get_label(): # label2id 根据标签取对应的索引及实体名称,id2label 根据索引取标签
    df = pd.read_csv(LABEL_PATH, names=['label', 'id'])
    return list(df['label']), dict(df.values)



"""定义Dataset类,将文本加载到DataFrame中"""

class Dataset(data.Dataset):

    def __init__(self, type='train', base_len=50): # type：区分加载的是训练集还是测试集;base_len:定义句子的参考长度
        super().__init__()
        self.base_len = base_len
        sample_path = TRAIN_SAMPLE_PATH if type == 'train' else TEST_SAMPLE_PATH
        self.df = pd.read_csv(sample_path, names=['word', 'label']) # 读取文件,格式[word,label]
        _, self.word2id = get_vocab() # 调用get_vocab函数,得到词表中字的索引列表
        _, self.label2id = get_label() # 调用get_label函数,得到标签表中标签的索引
        self.get_points()
        # 初始化Bert
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)


    """文本切分,50字切一刀"""

    def get_points(self):
        self.points = [0] # 初始切点
        i = 0 # 下一个即将要切的位置
        while True:
            if i + self.base_len >= len(self.df):
                self.points.append(len(self.df))
                break
            if self.df.loc[i + self.base_len, 'label'] == 'O':
                i += self.base_len
                self.points.append(i)
            else:
                i += 1

    """文本数字化(根据切点计算句子数)"""

    def __len__(self):
        return len(self.points) - 1
    

    """取单个句子,对单个句子中的字转id"""
    def __getitem__(self, index): # index:取第几段句子
        df = self.df[self.points[index]:self.points[index + 1]]
        word_unk_id = self.word2id[WORD_UNK] # 词表中字转 id
        label_o_id = self.label2id['O']

        # input：输入句子中每个字对应的位置索引组成的列表
        # target：输入句子中每个字对应的标签
        # input = [self.word2id.get(w, word_unk_id) for w in df['word']] # 遍历切中的句子中每个字在word2id中找到对应的数字索引,没有的话取unk
        # 注意：先自己将句子做分词，再转id，避免bert自动分词导致句子长度变化
        input = self.tokenizer.encode(list(df['word']), add_special_tokens=False)
        target = [self.label2id.get(l, label_o_id) for l in df['label']] # label也是一样的操作
        # return input, target
        # bert要求句子长度不能超过512
        return input[:MAX_POSITION_EMBEDDINGS], target[:MAX_POSITION_EMBEDDINGS]


"""数据校对整理,对句子长度进行排序,将短句子填充至最大句子的长度保持一致"""

def collate_fn(batch):
    # print(batch[3]) # batch为元组:五十个字对应的id和label对应的id
    # exit()
    batch.sort(key=lambda x: len(x[0]), reverse=True) # k=lambda x:取得是batch[0][1]这样的元素
    max_len = len(batch[0][0]) # 整个batch的最大长度
    input = []
    target = []
    mask = []
    for item in batch: # 遍历batch中每一个item
        pad_len = max_len - len(item[0]) # 填充长度
        input.append(item[0] + [WORD_PAD_ID] * pad_len)
        target.append(item[1] + [LABEL_O_ID] * pad_len)
        mask.append([1] * len(item[0]) + [0] * pad_len)
    return torch.tensor(input), torch.tensor(target), torch.tensor(mask).bool()


def extract(label, text):
    i = 0  
    res = []
    while i < len(label): # 判断当前标签是不是'O'
        if label[i] != 'O':
            prefix, name = label[i].split('-')
            start = end = i
            i += 1
            while i < len(label) and label[i] == 'I-' + name:
                end = i
                i += 1
            res.append([name, text[start:end + 1]])
        else:
            i += 1
    return res


def report(y_true, y_pred):
    return classification_report(y_true, y_pred, digits=4)
    

if __name__ == '__main__':

    # 调用标签表,得到标签到索引及索引到标签
    # id2label, label2id = get_label()
    # print(id2label, label2id)
    # exit()

    dataset = Dataset()

    # 加载Dataloader类
    loader = data.DataLoader(dataset, batch_size=16, collate_fn=collate_fn)
    print(next(iter(loader)))
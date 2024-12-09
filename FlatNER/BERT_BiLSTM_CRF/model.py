import torch.nn as nn
from config import *
from torchcrf import CRF
import torch
from transformers import BertModel

class Model(nn.Module): 
    def __init__(self):
        super().__init__()
       # self.embed = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM, WORD_PAD_ID)
        self.bert = BertModel.from_pretrained(BERT_MODEL)
        self.lstm = nn.LSTM(
            EMBEDDING_DIM,  
            HIDDEN_SIZE,
            batch_first=True, # 传入的数据在第一个维度表示每个batch的大小 
            bidirectional=True,
        )
        self.linear = nn.Linear(2 * HIDDEN_SIZE, TARGET_SIZE)
        self.crf = CRF(TARGET_SIZE, batch_first=True)

    def _get_lstm_feature(self, input, mask):
        # out = self.embed(input)
        out = self.bert(input, mask)[0]
        out, _ = self.lstm(out) # out:序列
        return self.linear(out)

    def forward(self, input, mask):
        out = self._get_lstm_feature(input, mask)
        return self.crf.decode(out, mask)

    def loss_fn(self, input, target, mask):
        y_pred = self._get_lstm_feature(input, mask)
        return -self.crf.forward(y_pred, target, mask, reduction='mean') # 每一个batch算一个平均损失

if __name__ == '__main__':
    model = Model()
    input = torch.randint(0, 3000, (100, 50)) # 每个batch取100个句子,每个句子有50个字

    # print(model)
    # exit()
     
    print(model(input, None)) # [100, 50, 31]:100每个batch是100个句子;50:每个句子长度为50;31：50个字中每个字对应31类的概率值
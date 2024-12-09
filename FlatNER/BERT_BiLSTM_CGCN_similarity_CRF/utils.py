# utils.py
import torch
from ltp import LTP
from torch.utils import data
from config import *
import pandas as pd
from seqeval.metrics import classification_report
from transformers import BertTokenizer
from transformers import logging
import fasttext

logging.set_verbosity_warning()

ltp = LTP(LTP_PATH)
fasttext_model = fasttext.load_model(FASTTEXT_MODEL_PATH)

def get_dependency_edges(text):
    result = ltp.pipeline([text], tasks=["cws", "dep"])
    edges = []
    heads = result.dep[0]['head']
    for mod, head in enumerate(heads):
        head = head - 1
        if head >= 0:
            edges.append((head, mod))
    return edges, result.cws[0]

def get_vocab():
    df = pd.read_csv(VOCAB_PATH, names=['word', 'id'])
    return list(df['word']), dict(df.values)

def get_label():
    df = pd.read_csv(LABEL_PATH, names=['label', 'id'])
    return list(df['label']), dict(df.values)

class Dataset(data.Dataset):
    def __init__(self, type='train', base_len=50):
        super().__init__()
        self.base_len = base_len
        sample_path = TRAIN_SAMPLE_PATH if type == 'train' else TEST_SAMPLE_PATH
        self.df = pd.read_csv(sample_path, names=['word', 'label'])
        _, self.word2id = get_vocab()
        _, self.label2id = get_label()
        self.get_points()
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    def get_points(self):
        self.points = [0]
        i = 0
        while True:
            if i + self.base_len >= len(self.df):
                self.points.append(len(self.df))
                break
            if self.df.loc[i + self.base_len, 'label'] == 'O':
                i += self.base_len
                self.points.append(i)
            else:
                i += 1

    def __len__(self):
        return len(self.points) - 1

    def __getitem__(self, index):
        df = self.df[self.points[index]:self.points[index + 1]]
        word_unk_id = self.word2id[WORD_UNK]
        label_o_id = self.label2id['O']

        input = self.tokenizer.encode(list(df['word'].str.replace('\n', '').str.replace('\r', '')), add_special_tokens=False)
        target = [self.label2id.get(l, label_o_id) for l in df['label']]

        text = ''.join(df['word'].str.replace('\n', '').str.replace('\r', ''))
        dependency_edges, words = get_dependency_edges(text)

        word_vectors = torch.stack([torch.tensor(fasttext_model.get_word_vector(word)) for word in words])

        sequence_edges = [(i, i + 1) for i in range(len(input) - 1)]

        return input[:MAX_POSITION_EMBEDDINGS], target[:MAX_POSITION_EMBEDDINGS], sequence_edges, dependency_edges, word_vectors, text

def collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    max_input_len = len(batch[0][0])
    word_vector_dim = batch[0][4].size(1)
    
    input, target, mask = [], [], []
    all_sequence_edges, all_dependency_edges, all_dependency_masks = [], [], []
    all_word_vectors, all_word_vectors_masks, all_texts = [], [], []

    for item in batch:
        pad_len = max_input_len - len(item[0])
        input.append(item[0] + [WORD_PAD_ID] * pad_len)
        target.append(item[1] + [LABEL_O_ID] * pad_len)
        mask.append([1] * len(item[0]) + [0] * pad_len)

        sequence_edges = [(i, j) for i, j in item[2] if i < max_input_len and j < max_input_len]
        all_sequence_edges.append(sequence_edges)

        dependency_edges = [(i, j) for i, j in item[3] if i < max_input_len and j < max_input_len]
        padding_len = max(0, len(sequence_edges) - len(dependency_edges))
        dependency_edges.extend([(-1, -1)] * padding_len)
        all_dependency_edges.append(dependency_edges)

        dependency_mask = [1] * len(dependency_edges) + [0] * (max_input_len - len(dependency_edges))
        all_dependency_masks.append(dependency_mask[:max_input_len])

        padded_word_vectors = torch.zeros((max_input_len, word_vector_dim))
        actual_len = item[4].size(0)
        padded_word_vectors[:actual_len, :] = item[4]
        all_word_vectors.append(padded_word_vectors)

        word_vectors_mask = [1] * actual_len + [0] * (max_input_len - actual_len)
        all_word_vectors_masks.append(word_vectors_mask)
        all_texts.append(item[5])

    return (
        torch.tensor(input),
        torch.tensor(target),
        torch.tensor(mask).bool(),
        all_sequence_edges,
        all_dependency_edges,
        torch.tensor(all_dependency_masks).bool(),
        torch.stack(all_word_vectors),
        torch.tensor(all_word_vectors_masks).bool(),
        all_texts
    )

def extract(label, text):
    i = 0  
    res = []
    while i < len(label):
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

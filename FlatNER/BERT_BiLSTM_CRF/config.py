import os
ORIGIN_DIR = '/home/deng/Maping/Graduation_experiment/Flat_NER/BERT_Pytorch_BiLSTM_CRF_NER/input/origin/'
ANNOTATION_DIR = '/home/deng/Maping/Graduation_experiment/Flat_NER/BERT_Pytorch_BiLSTM_CRF_NER/output/annotation/'

TRAIN_SAMPLE_PATH = '/home/deng/Maping/Graduation_experiment/Flat_NER/BERT_Pytorch_BiLSTM_CRF_NER/output/dataset/Ruijin/train_sample.txt'
TEST_SAMPLE_PATH = '/home/deng/Maping/Graduation_experiment/Flat_NER/BERT_Pytorch_BiLSTM_CRF_NER/output/dataset/Ruijin/test_sample.txt'

VOCAB_PATH = '/home/deng/Maping/Graduation_experiment/Flat_NER/BERT_Pytorch_BiLSTM_CRF_NER/output/dataset/Ruijin/vocab.txt'
LABEL_PATH = '/home/deng/Maping/Graduation_experiment/Flat_NER/BERT_Pytorch_BiLSTM_CRF_NER/output/dataset/Ruijin/label.txt'

WORD_PAD = '<PAD>'
WORD_UNK = '<UNK>'

WORD_PAD_ID = 0
WORD_UNK_ID = 1
LABEL_O_ID = 0

VOCAB_SIZE = 3000
EMBEDDING_DIM = 100
HIDDEN_SIZE = 128
TARGET_SIZE = 31
LR = 1e-5
EPOCH = 50

MODEL_DIR = '/home/deng/Maping/Graduation_experiment/Flat_NER/BERT_Pytorch_BiLSTM_CRF_NER/output/model/Ruijin/'
LOG_PATH = '/home/deng/Maping/Graduation_experiment/Flat_NER/BERT_Pytorch_BiLSTM_CRF_NER/Logging/Ruijin.log'

import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Bert 
BERT_MODEL = '/home/deng/Maping/Graduation_experiment/Flat_NER/BERT_Pytorch_BiLSTM_CRF_NER/HuggingFace_Model/bert-base-chinese'
EMBEDDING_DIM = 768
MAX_POSITION_EMBEDDINGS = 512
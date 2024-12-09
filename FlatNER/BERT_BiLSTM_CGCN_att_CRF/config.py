import os
import torch

# 数据路径配置
ORIGIN_DIR = '/home/user/swun/Maping/BERT_Pytorch_BiLSTM_CRF_NER/input/origin/'
ANNOTATION_DIR = '/home/user/swun/Maping/BERT_Pytorch_BiLSTM_CRF_NER/output/annotation/'
TRAIN_SAMPLE_PATH = '/home/deng/Maping/Graph_experiment/BERT_BiLSTM_GCN_CRF/output/dataset/Ruijin_EDA/train_sample.txt'
TEST_SAMPLE_PATH = '/home/deng/Maping/Graph_experiment/BERT_BiLSTM_GCN_CRF/output/dataset/Ruijin_EDA/test_sample.txt'
VOCAB_PATH = '/home/deng/Maping/Graph_experiment/BERT_BiLSTM_GCN_CRF/output/dataset/Ruijin_EDA/vocab.txt'
LABEL_PATH = '/home/deng/Maping/Graph_experiment/BERT_BiLSTM_GCN_CRF/output/dataset/Ruijin_EDA/label.txt'
MODEL_DIR = '/home/deng/Maping/Graph_experiment/BERT_BiLSTM_GCN_CRF/output/model/BERT_BiLSTM_GCN_LN_CRF/'
LTP_PATH = '/home/deng/Maping/Graph_experiment/BERT_BiLSTM_GCN_CRF/HuggingFace_Model/LTP_BASE1'
FASTTEXT_MODEL_PATH = '/home/deng/Maping/Graph_experiment/BERT_BiLSTM_GCN_CRF/HuggingFace_Model/Fasttext_model/medical_word_vector_model.bin'

# 特殊标记与ID
WORD_PAD = '<PAD>'
WORD_UNK = '<UNK>'
WORD_PAD_ID = 0
WORD_UNK_ID = 1
LABEL_O_ID = 0

# 模型参数
VOCAB_SIZE = 3000  # 可根据实际词汇表大小调整
EMBEDDING_DIM = 768  # BERT-base 的嵌入维度
HIDDEN_SIZE = 128
TARGET_SIZE = 31  # 分类标签的数量
MAX_POSITION_EMBEDDINGS = 512  # BERT 的最大位置嵌入

# 训练超参数
LR = 1e-5  # 总学习率
EPOCH = 50  # 训练轮次

# 设备配置
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# BERT 模型路径
BERT_MODEL = '/home/deng/Maping/Graph_experiment/BERT_BiLSTM_GCN_CRF/HuggingFace_Model/bert-base-chinese'

# main.py
import logging as py_logging
import torch
from torch import nn
from torch.utils import data
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report
from utils import *
from model import *
from config import *
from ltp import LTP
import os
import warnings
warnings.filterwarnings("ignore")
writer = SummaryWriter(log_dir='/home/deng/Maping/Graph_experiment/BERT_BiLSTM_GCN_CRF/Tensorboard/')

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ltp = LTP(LTP_PATH)

py_logging.basicConfig(
    level=py_logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        py_logging.FileHandler("/home/deng/Maping/Graph_experiment/BERT_BiLSTM_GCN_CRF/Logging/BERT_BiLSTM_CGCN_att_BN_CRF.log"),
        py_logging.StreamHandler()
    ]
)

def calculate_f1_scores(y_true, y_pred):
    report = classification_report(y_true, y_pred, digits=4, output_dict=True)
    return {
        'macro_f1': report['macro avg']['f1-score'],
        'micro_f1': report['micro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score']
    }

def get_optimizer_with_warmup_and_layerwise_lr(model, base_lr, bert_lr, lstm_lr, crf_lr, warmup_steps, total_steps):
    no_decay = ['bias', 'LayerNorm.weight']
    
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01, 'lr': bert_lr
        },
        {
            'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0, 'lr': bert_lr
        },
        {
            'params': [p for n, p in model.lstm.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01, 'lr': lstm_lr
        },
        {
            'params': model.crf.parameters(),
            'weight_decay': 0.01, 'lr': crf_lr
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=base_lr)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler

def train_and_evaluate(epochs, model_dir, device):
    writer = SummaryWriter()

    dataset = Dataset()
    loader = data.DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=16)
    model = Model(ltp).to(device)

    base_lr = 1e-5
    bert_lr = 1e-5
    lstm_lr = 2e-5
    crf_lr = 2e-5
    warmup_steps = 1000
    total_steps = len(loader) * epochs
    
    optimizer, scheduler = get_optimizer_with_warmup_and_layerwise_lr(model, base_lr, bert_lr, lstm_lr, crf_lr, warmup_steps, total_steps)

    best_micro_f1 = 0.0

    for e in range(epochs):
        model.train()
        total_loss = 0.0

        pbar = tqdm(total=len(loader), desc=f'Training Epoch {e+1}/{epochs}')

        for batch in loader:
            input, target, mask, sequence_edges, dependency_edges, dependency_masks, word_vectors, word_vectors_masks, input_texts = batch
            input, target, mask = input.to(device), target.to(device), mask.to(device)
            word_vectors = word_vectors.to(device)
            dependency_masks = dependency_masks.to(device)

            dependency_edges_hop1 = [model._sample_subgraph(dep, hop=1, sample_type='default') for dep in dependency_edges]
            dependency_edges_hop2 = [
                model._sample_subgraph(
                    dep,
                    embeddings=word_vectors[batch_idx],
                    hop=2,
                    sample_type='attention',
                    top_k=3,
                    valid_mask=word_vectors_masks[batch_idx]
                )
                for batch_idx, dep in enumerate(dependency_edges)
            ]

            loss = model.loss_fn(
                input, target, mask, sequence_edges, dependency_edges_hop1,
                dependency_edges_hop2, word_vectors, input_texts,
                dependency_masks, word_vectors_masks
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            writer.add_scalar('Loss/train', loss.item(), e * len(loader) + pbar.n)
            pbar.update(1)

        pbar.close()

        avg_loss = total_loss / len(loader)
        writer.add_scalar('Loss/train_epoch', avg_loss, e)
        py_logging.info(f'>> Epoch: {e+1}, Average Loss: {avg_loss:.4f}')

        model.eval()
        test_loss, y_true_list, y_pred_list = evaluate(model, device)

        f1_scores = calculate_f1_scores(y_true_list, y_pred_list)
        writer.add_scalar('Loss/test', test_loss, e)
        writer.add_scalar('F1/macro', f1_scores['macro_f1'], e)
        writer.add_scalar('F1/micro', f1_scores['micro_f1'], e)
        writer.add_scalar('F1/weighted', f1_scores['weighted_f1'], e)

        if f1_scores['micro_f1'] > best_micro_f1:
            best_micro_f1 = f1_scores['micro_f1']
            best_model_path = os.path.join(model_dir, f'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            py_logging.info(f"Best model saved at: {best_model_path}")

        py_logging.info(f'>> Epoch: {e+1}, Test Loss: {test_loss:.4f}, Macro F1: {f1_scores["macro_f1"]:.4f}, Micro F1: {f1_scores["micro_f1"]:.4f}, Weighted F1: {f1_scores["weighted_f1"]:.4f}')

    writer.close()
    return best_model_path

def evaluate(model, device):
    dataset = Dataset('test')
    loader = data.DataLoader(dataset, batch_size=16, collate_fn=collate_fn, num_workers=16)
 
    y_true_list = []
    y_pred_list = []
    total_loss = 0.0

    id2label, _ = get_label()

    with torch.no_grad():
        pbar = tqdm(total=len(loader), desc='Evaluating')
        for batch in loader:
            input, target, mask, sequence_edges, dependency_edges, dependency_masks, word_vectors, word_vectors_masks, input_texts = batch
            input, target, mask = input.to(device), target.to(device), mask.to(device)
            word_vectors = word_vectors.to(device)
            dependency_masks = dependency_masks.to(device)

            dependency_edges_hop1 = [model._sample_subgraph(dep, hop=1, sample_type='default') for dep in dependency_edges]
            dependency_edges_hop2 = [
                model._sample_subgraph(
                    dep,
                    embeddings=word_vectors[batch_idx],
                    hop=2,
                    sample_type='attention',
                    top_k=3,
                    valid_mask=word_vectors_masks[batch_idx]
                )
                for batch_idx, dep in enumerate(dependency_edges)
            ]

            y_pred = model(input, mask, sequence_edges, dependency_edges_hop1, dependency_edges_hop2, word_vectors, input_texts, dependency_masks, word_vectors_masks)

            loss = model.loss_fn(input, target, mask, sequence_edges, dependency_edges_hop1, dependency_edges_hop2, word_vectors, input_texts, dependency_masks, word_vectors_masks)
            total_loss += loss.item()

            for lst in y_pred:
                lst = lst if isinstance(lst, list) else lst.tolist()
                y_pred_list.append([id2label[i] for i in lst])

            for y, m in zip(target, mask):
                y_true_list.append([id2label[i] for i in y[m == True].tolist()])
            pbar.update(1)

    avg_loss = total_loss / len(loader)

    # 打印分类报告
    report_str = classification_report(y_true_list, y_pred_list, digits=4)
    py_logging.info(report_str)

    return avg_loss, y_true_list, y_pred_list

if __name__ == '__main__':
    best_model_path = train_and_evaluate(EPOCH, MODEL_DIR, DEVICE)
    model = Model(ltp)
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.to(DEVICE)
    
    avg_loss, y_true_list, y_pred_list = evaluate(model, DEVICE)
    final_f1_scores = calculate_f1_scores(y_true_list, y_pred_list)
    py_logging.info(f'Final Evaluation - Test Loss: {avg_loss:.4f}, Macro F1: {final_f1_scores["macro_f1"]:.4f}, Micro F1: {final_f1_scores["micro_f1"]:.4f}, Weighted F1: {final_f1_scores["weighted_f1"]:.4f}')

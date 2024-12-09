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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 设置日志
py_logging.basicConfig(
    level=py_logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        py_logging.FileHandler("/home/deng/Maping/Graduation_experiment/Flat_NER/BERT_Pytorch_BiLSTM_CRF_NER/Logging/EDA_CHD_training.log"),
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

def train_and_evaluate(epochs, model_dir, device):
    # TensorBoard设置
    writer = SummaryWriter()

    dataset = Dataset()
    loader = data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )

    model = Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_micro_f1 = 0.0  # 初始化最佳micro平均f1分数
    best_model_path = None

    for e in range(epochs):
        model.train()
        total_loss = 0.0

        # 使用 tqdm 包装 DataLoader 以显示进度条，只显示 epoch 进度
        pbar = tqdm(total=len(loader), desc=f'Training Epoch {e+1}/{epochs}')

        for b, (input, target, mask) in enumerate(loader):
            input = input.to(device)
            mask = mask.to(device)
            target = target.to(device)

            y_pred = model(input, mask)
            loss = model.loss_fn(input, target, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 记录每个batch的损失到TensorBoard
            writer.add_scalar('Loss/train', loss.item(), e * len(loader) + b)

            # 更新进度条
            pbar.update(1)

        pbar.close()  # 结束epoch后关闭进度条

        avg_loss = total_loss / len(loader)
        writer.add_scalar('Loss/train_epoch', avg_loss, e)
        py_logging.info(f'>> Epoch: {e+1}, Average Loss: {avg_loss:.4f}')  # 记录每个epoch的平均损失

        # 每个epoch后评估模型
        model.eval()
        test_loss, y_true_list, y_pred_list = evaluate(model, device)

        # 计算各类F1分数
        f1_scores = calculate_f1_scores(y_true_list, y_pred_list)
        
        # 记录测试损失和F1分数到TensorBoard
        writer.add_scalar('Loss/test', test_loss, e)
        writer.add_scalar('F1/macro', f1_scores['macro_f1'], e)
        writer.add_scalar('F1/micro', f1_scores['micro_f1'], e)
        writer.add_scalar('F1/weighted', f1_scores['weighted_f1'], e)

        # 检查是否是最佳模型
        if f1_scores['micro_f1'] > best_micro_f1:
            best_micro_f1 = f1_scores['micro_f1']
            best_model_path = model_dir + f'best_model.pth'
            torch.save(model, best_model_path)
            py_logging.info(f"Best model saved at: {best_model_path}")

        # 记录测试损失和评估指标
        py_logging.info(f'>> Epoch: {e+1}, Test Loss: {test_loss:.4f}, Macro F1: {f1_scores["macro_f1"]:.4f}, Micro F1: {f1_scores["micro_f1"]:.4f}, Weighted F1: {f1_scores["weighted_f1"]:.4f}')

    writer.close()
    return best_model_path


def evaluate(model, device):
    dataset = Dataset('test')
    loader = data.DataLoader(dataset, batch_size=8, collate_fn=collate_fn,num_workers=4)

    y_true_list = []
    y_pred_list = []
    total_loss = 0.0

    id2label, _ = get_label()

    with torch.no_grad():
        # 使用 tqdm 包装 DataLoader 以显示进度条
        with tqdm(total=len(loader), desc='Evaluating') as pbar:
            for b, (input, target, mask) in enumerate(loader):
                input = input.to(device)
                mask = mask.to(device)
                target = target.to(device)

                y_pred = model(input, mask)
                loss = model.loss_fn(input, target, mask)
                total_loss += loss.item()

                # 拼接返回值
                for lst in y_pred:
                    y_pred_list.append([id2label[i] for i in lst])
                for y, m in zip(target, mask):
                    y_true_list.append([id2label[i] for i in y[m == True].tolist()])

                # 更新进度条
                pbar.update(1)

    avg_loss = total_loss / len(loader)

    # 打印报告
    report_str = classification_report(y_true_list, y_pred_list, digits=4)
    py_logging.info(report_str)

    return avg_loss, y_true_list, y_pred_list

if __name__ == '__main__':
    best_model_path = train_and_evaluate(EPOCH, MODEL_DIR, DEVICE)

    # 加载最佳模型进行测试
    model = torch.load(best_model_path, map_location=DEVICE)
    py_logging.info("Best model loaded for final evaluation.")
    avg_loss, y_true_list, y_pred_list = evaluate(model, DEVICE)

    # 计算并记录最终的评估指标
    final_f1_scores = calculate_f1_scores(y_true_list, y_pred_list)
    py_logging.info(f'Final Evaluation - Test Loss: {avg_loss:.4f}, Macro F1: {final_f1_scores["macro_f1"]:.4f}, Micro F1: {final_f1_scores["micro_f1"]:.4f}, Weighted F1: {final_f1_scores["weighted_f1"]:.4f}')

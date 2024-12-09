"""
Date: 2021-05-31 19:50:58
LastEditors: GodK
"""
import numpy as np
import os
import config
import sys
import torch
import json
from transformers import BertTokenizerFast, BertModel
from common.utils import Preprocessor, multilabel_categorical_crossentropy
from models.GlobalPointer import DataMaker, MyDataset, GlobalPointer, MetricsCalculator
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import glob
import wandb
from evaluate import load_model
import time
from sklearn.metrics import classification_report, precision_recall_fscore_support
import logging


config = config.train_config
hyper_parameters = config["hyper_parameters"]

os.environ["TOKENIZERS_PARALLELISM"] = "true"
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
config["num_workers"] = 6 if sys.platform.startswith("linux") else 0

# for reproductivity
torch.manual_seed(hyper_parameters["seed"])  # pytorch random seed
torch.backends.cudnn.deterministic = True

# Initialize wandb
if config["logger"] == "wandb" and config["run_type"] == "train":
    wandb.init(project="GlobalPointer_pytorch_" + config["exp_name"], config=hyper_parameters)
    wandb.run.name = config["run_name"] + "_" + wandb.run.id
    model_state_dict_dir = wandb.run.dir
    logger = wandb
elif config["run_type"] == "train":
    model_state_dict_dir = os.path.join(config["path_to_save_model"], config["exp_name"],
                                        time.strftime("%Y-%m-%d_%H.%M.%S", time.gmtime()))
    if not os.path.exists(model_state_dict_dir):
        os.makedirs(model_state_dict_dir)


tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"], add_special_tokens=True, do_lower_case=False)


def load_data(data_path, data_type="train"):
    """读取数据集

    Args:
        data_path (str): 数据存放路径
        data_type (str, optional): 数据类型. Defaults to "train".

    Returns:
        (json): train和valid中一条数据格式：{"text":"","entity_list":[(start, end, label), (start, end, label)...]}
    """
    if data_type == "train" or data_type == "valid":
        datas = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                item = {}
                item["text"] = line["text"]
                item["entity_list"] = []
                for k, v in line['label'].items():
                    for spans in v.values():
                        for start, end in spans:
                            item["entity_list"].append((start, end, k))
                datas.append(item)
        return datas
    else:
        return json.load(open(data_path, encoding="utf-8"))


ent2id_path = os.path.join(config["data_home"], config["exp_name"], config["ent2id"])
ent2id = load_data(ent2id_path, "ent2id")
ent_type_size = len(ent2id)


def data_generator(data_type="train"):
    """
    读取数据，生成DataLoader。
    """

    if data_type == "train":
        train_data_path = os.path.join(config["data_home"], config["exp_name"], config["train_data"])
        train_data = load_data(train_data_path, "train")
        valid_data_path = os.path.join(config["data_home"], config["exp_name"], config["valid_data"])
        valid_data = load_data(valid_data_path, "valid")
    elif data_type == "valid":
        valid_data_path = os.path.join(config["data_home"], config["exp_name"], config["valid_data"])
        valid_data = load_data(valid_data_path, "valid")
        train_data = []
    elif data_type == "test":
        valid_data_path = os.path.join(config["data_home"], config["exp_name"], config["test_data"])
        valid_data = load_data(valid_data_path, "valid")
        train_data = []

    all_data = train_data + valid_data

    # TODO:句子截取
    max_tok_num = 0
    problematic_sample = None
    for sample in all_data:
        tokens = tokenizer(sample["text"])["input_ids"]
        if len(tokens) > max_tok_num:
            max_tok_num = len(tokens)
            problematic_sample = sample["text"]
    assert max_tok_num <= hyper_parameters[
        "max_seq_len"], f'数据文本最大token数量{max_tok_num}超过预设{hyper_parameters["max_seq_len"]}'
    max_seq_len = min(max_tok_num, hyper_parameters["max_seq_len"])



    data_maker = DataMaker(tokenizer)

    if data_type == "train":
        # train_inputs = data_maker.generate_inputs(train_data, max_seq_len, ent2id)
        # valid_inputs = data_maker.generate_inputs(valid_data, max_seq_len, ent2id)
        train_dataloader = DataLoader(MyDataset(train_data),
                                      batch_size=hyper_parameters["batch_size"],
                                      shuffle=True,
                                      num_workers=config["num_workers"],
                                      drop_last=False,
                                      collate_fn=lambda x: data_maker.generate_batch(x, max_seq_len, ent2id)
                                      )
        valid_dataloader = DataLoader(MyDataset(valid_data),
                                      batch_size=hyper_parameters["batch_size"],
                                      shuffle=True,
                                      num_workers=config["num_workers"],
                                      drop_last=False,
                                      collate_fn=lambda x: data_maker.generate_batch(x, max_seq_len, ent2id)
                                      )
        # for batch in train_dataloader:
        #     print(batch[1].shape)
        #     print(hyper_parameters["batch_size"])
        #     break
        return train_dataloader, valid_dataloader
    else:
        # valid_inputs = data_maker.generate_inputs(valid_data, max_seq_len, ent2id)
        valid_dataloader = DataLoader(MyDataset(valid_data),
                                      batch_size=hyper_parameters["batch_size"],
                                      shuffle=True,
                                      num_workers=config["num_workers"],
                                      drop_last=False,
                                      collate_fn=lambda x: data_maker.generate_batch(x, max_seq_len, ent2id)
                                      )
        return valid_dataloader


metrics = MetricsCalculator()


def train_step(batch_train, model, optimizer, criterion):
    batch_samples, batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = batch_train
    batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = (
        batch_input_ids.to(device),
        batch_attention_mask.to(device),
        batch_token_type_ids.to(device),
        batch_labels.to(device)
    )
    logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
    loss = criterion(batch_labels, logits)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()




encoder = BertModel.from_pretrained(config["bert_path"])
model = GlobalPointer(encoder, ent_type_size, 64)
model = model.to(device)

if config["logger"] == "wandb" and config["run_type"] == "train":
    wandb.watch(model)


def train(model, dataloader, epoch, optimizer):
    model.train()

    # 定义损失函数
    def loss_fun(y_true, y_pred):
        """
        y_true:(batch_size, ent_type_size, seq_len, seq_len)
        y_pred:(batch_size, ent_type_size, seq_len, seq_len)
        """
        batch_size, ent_type_size = y_pred.shape[:2]
        y_true = y_true.reshape(batch_size * ent_type_size, -1)
        y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
        loss = multilabel_categorical_crossentropy(y_true, y_pred)
        return loss
    
    
    # scheduler
    if hyper_parameters["scheduler"] == "CAWR":
        T_mult = hyper_parameters["T_mult"]
        rewarm_epoch_num = hyper_parameters["rewarm_epoch_num"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                         len(train_dataloader) * rewarm_epoch_num,
                                                                         T_mult)
    elif hyper_parameters["scheduler"] == "Step":
        decay_rate = hyper_parameters["decay_rate"]
        decay_steps = hyper_parameters["decay_steps"]
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=decay_rate)
    else:
        scheduler = None

    total_loss = 0.0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for batch_ind, batch_data in pbar:
        loss = train_step(batch_data, model, optimizer, loss_fun)
        total_loss += loss
        avg_loss = total_loss / (batch_ind + 1)
        if scheduler is not None:
            scheduler.step()
        pbar.set_description(f"Epoch {epoch + 1}/{hyper_parameters['epochs']}")
        pbar.set_postfix(loss=avg_loss, lr=optimizer.param_groups[0]["lr"])
    if config["logger"] == "wandb":
        logger.log({"Train Loss": avg_loss, "epoch": epoch + 1})
    return avg_loss




def valid_step(batch_valid, model):
    batch_samples, batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = batch_valid
    batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = (
        batch_input_ids.to(device),
        batch_attention_mask.to(device),
        batch_token_type_ids.to(device),
        batch_labels.to(device)
    )

    with torch.no_grad():
        logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
    # 将 logits 转换为预测值
    predictions = torch.sigmoid(logits) > 0.5
    predictions = predictions.cpu().numpy()
    batch_labels = batch_labels.cpu().numpy()

    return batch_labels, predictions


def valid(model, dataloader, epoch):
    model.eval()
    all_labels = []
    all_predictions = []
    for batch_data in tqdm(dataloader, desc="Validating"):
        # 获取验证数据的预测值和标签
        batch_labels, batch_predictions = valid_step(batch_data, model)  # 不再处理损失值
        all_labels.append(batch_labels)
        all_predictions.append(batch_predictions)
    
    # 计算全局指标
    all_labels = np.concatenate(all_labels, axis=0).reshape(-1, ent_type_size)
    all_predictions = np.concatenate(all_predictions, axis=0).reshape(-1, ent_type_size)
    report = classification_report(all_labels, all_predictions, target_names=list(ent2id.keys()), digits=4, output_dict=True)
    micro_f1 = report["micro avg"]["f1-score"]
    precision = report["micro avg"]["precision"]
    recall = report["micro avg"]["recall"]
    formatted_report = classification_report(all_labels, all_predictions, target_names=list(ent2id.keys()), digits=4)
    print(formatted_report)
    
    # Log metrics to wandb
    if config["logger"] == "wandb":
        logger.log({
            "Validation F1 Score": micro_f1,
            "Validation Precision": precision,
            "Validation Recall": recall
        })
        # Record the formatted classification report
        table = wandb.Table(columns=["Report"], data=[[line] for line in formatted_report.split("\n")])
        logger.log({"Validation Classification Report": table})

    return report, formatted_report


if __name__ == '__main__':
    if config["run_type"] == "train":
        train_dataloader, valid_dataloader = data_generator()

        # 定义优化器
        init_learning_rate = float(hyper_parameters["lr"])
        optimizer = torch.optim.Adam(model.parameters(), lr=init_learning_rate)

        max_f1 = 0.0
        best_epoch = 0 
        for epoch in range(hyper_parameters["epochs"]):
            # 训练阶段
            train_loss = train(model, train_dataloader, epoch, optimizer)
            
            # 验证阶段
            valid_report, formatted_report = valid(model, valid_dataloader, epoch)
            valid_f1 = valid_report["micro avg"]["f1-score"]

            # 记录日志
            logging.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}")
            logging.info(f"\nValidation Report:\n{formatted_report}")
            print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Valid F1: {valid_f1:.4f}")

            # 保存模型
            if valid_f1 > max_f1:
                max_f1 = valid_f1
                best_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(model_state_dict_dir, f"model_state_{epoch + 1}.pt"))
                print(f"New best model saved at epoch {best_epoch} with F1: {max_f1:.8f}")
                logging.info(f"New best model saved at epoch {best_epoch} with F1: {max_f1:.8f}")
            logging.info(f"Epoch {epoch + 1} - Best Valid F1: {max_f1:.8f}")

        # 打印最终最佳 F1 值和对应的轮次
        print(f"Training complete. Best Valid F1: {max_f1:.8f} at epoch {best_epoch}.")
        logging.info(f"Training complete. Best Valid F1: {max_f1:.8f} at epoch {best_epoch}.")

    elif config["run_type"] == "eval":
        # 此处的 eval 是为了评估测试集的 p r f1（如果测试集有标签的情况），无标签预测使用 evaluate.py
        model = load_model()
        test_dataloader = data_generator(data_type="test")
        valid(model, test_dataloader)



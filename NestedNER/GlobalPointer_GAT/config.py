import os
import time

# 获取当前脚本所在的根目录
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# 通用配置
common = {
    "exp_name": "CHD",
    "encoder": "BERT",
    "data_home": os.path.join(BASE_DIR, "datasets"),
    "bert_path": os.path.join(BASE_DIR, "pretrained_models", "bert-base-chinese"),  # 相对路径拼接
    "run_type": "train",  # train, eval
    "f1_2_save": 0.5,  # 存模型的最低f1值
    "logger": "wandb"  # wandb or default，default意味着只输出日志到控制台
}

# wandb的配置
wandb_config = {
    "run_name": time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
    "log_interval": 10
}

# 训练配置
train_config = {
    "train_data": os.path.join(BASE_DIR, "datasets", "CHD", "train.json"),
    "valid_data": os.path.join(BASE_DIR, "datasets", "CHD", "dev.json"),
    "test_data": os.path.join(BASE_DIR, "datasets", "CHD", "dev.json"),
    "ent2id": os.path.join(BASE_DIR, "datasets", "CHD", "ent2id.json"),
    "path_to_save_model": os.path.join(BASE_DIR, "outputs/"),
    "hyper_parameters": {
        "lr": 2e-5,
        "batch_size": 32,
        "epochs": 50,
        "seed": 2333,
        "max_seq_len": 512,
        "scheduler": "CAWR"  # CAWR, Step, None
    }
}

# 评估配置
eval_config = {
    "model_state_dir": os.path.join(BASE_DIR, "outputs/"),  # 预测时填写模型路径
    "run_id": "",
    "last_k_model": 1,
    "predict_data": os.path.join(BASE_DIR, "datasets", "CHD", "test.json"),
    "ent2id": os.path.join(BASE_DIR, "datasets", "CHD", "ent2id.json"),
    "save_res_dir": os.path.join(BASE_DIR, "results"),
    "hyper_parameters": {
        "batch_size": 32,
        "max_seq_len": 512,
    }
}

# CosineAnnealingWarmRestarts 调度器配置
cawr_scheduler = {
    "T_mult": 1,
    "rewarm_epoch_num": 2,
}

# StepLR 调度器配置
step_scheduler = {
    "decay_rate": 0.999,
    "decay_steps": 200,
}

# 合并配置
train_config["hyper_parameters"].update(**cawr_scheduler, **step_scheduler)
train_config = {**train_config, **common, **wandb_config}
eval_config = {**eval_config, **common}

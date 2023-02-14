import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from trajdata import AgentBatch, AgentType, UnifiedDataset
from trajdata.visualization.vis import plot_agent_batch

from src.models import LSTM_Model
from src.utils.definitions import DATA_DIR, LOGS_DIR
from src.utils.draw import draw


def read_data(filename: str):
    data = []
    with open(filename) as f:
        for line in f:
            sample = list(map(float, line.split(" ")))
            data.append(sample)
    return data


class TrajDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (
            torch.tensor(self.data[index])[..., None],
            torch.tensor(self.targets[index])[..., None],
        )


if __name__ == "__main__":
    data = read_data(os.path.join(DATA_DIR, "simple/data"))
    target = read_data(os.path.join(DATA_DIR, "simple/pred"))
    train_data, test_data, train_label, test_label = train_test_split(
        data, target, test_size=0.2, random_state=1
    )
    train_data, val_data, train_label, val_label = train_test_split(
        train_data, train_label, test_size=0.2, random_state=1
    )

    BATCH_SIZE = 2
    PRED_NUM = len(target[0])
    train_data = TrajDataset(train_data, train_label)
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=train_sampler)

    val_data = TrajDataset(val_data, val_label)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    test_data = TrajDataset(test_data, test_label)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    time = datetime.now().strftime("%b%d_%H-%M-%S")
    writer = SummaryWriter(log_dir=os.path.join(LOGS_DIR, f"test_data/{time}/"))
    FEATURE_NUM = 1
    rnn_model = LSTM_Model(PRED_NUM, FEATURE_NUM, writer)
    # early_stop_callback = EarlyStopping(
    #     monitor="val_loss",
    #     min_delta=0.0,
    #     patience=1,
    #     verbose=True,
    #     mode="min"
    # )

    trainer = Trainer(
        gpus=0,
        # checkpoint_callback=False,
        accumulate_grad_batches=1,
        max_epochs=50,
        # progress_bar_refresh_rate=10,
    )  # callbacks=[early_stop_callback])

    trainer.fit(rnn_model, train_loader, val_loader)
    trainer.test(rnn_model, test_loader)

    writer.close()

    inputs, labels = next(iter(test_loader))
    _, results = rnn_model(inputs, labels)
    draw(inputs, labels, results)

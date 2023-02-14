import os
from datetime import datetime

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from trajdata import AgentType, UnifiedDataset

from src.models import LSTM_Model
from src.utils.definitions import LOGS_DIR

if __name__ == "__main__":
    train_dataset = UnifiedDataset(
        desired_data=["nusc_mini", "mini_train"],
        centric="agent",
        desired_dt=0.1,
        history_sec=(3.2, 3.2),
        future_sec=(4.8, 4.8),
        only_predict=[AgentType.PEDESTRIAN],
        # rebuild_cache=True,
        # rebuild_maps=True,
        num_workers=os.cpu_count(),
        verbose=True,
        data_dirs={"nusc_mini": "~/datasets/nuscenes"},
    )

    test_data = UnifiedDataset(
        desired_data=["nusc_mini", "mini_val"],
        centric="agent",
        desired_dt=0.1,
        history_sec=(3.2, 3.2),
        future_sec=(4.8, 4.8),
        only_predict=[AgentType.PEDESTRIAN],
        # rebuild_cache=True,
        # rebuild_maps=True,
        num_workers=os.cpu_count(),
        verbose=True,
        data_dirs={"nusc_mini": "~/datasets/nuscenes"},
    )

    DATASET_LEN = len(train_dataset)
    VAL_LEN = int(0.2 * DATASET_LEN)
    TRAIN_LEN = DATASET_LEN - VAL_LEN
    print(f"Total Data Samples: {len(train_dataset):,}")
    val_data, train_data = torch.utils.data.random_split(
        train_dataset, [VAL_LEN, TRAIN_LEN]
    )

    BATCH_SIZE = 64

    train_dataloader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=train_dataset.get_collate_fn(),
        num_workers=os.cpu_count(),
    )

    val_dataloader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=train_dataset.get_collate_fn(),
        num_workers=os.cpu_count(),
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=train_dataset.get_collate_fn(),
        num_workers=os.cpu_count(),
    )

    batch = next(iter(train_dataloader))
    print(
        "Shapes:", batch.agent_hist.shape, batch.agent_fut.shape
    )  # [batch_sz, seq_len, dim]

    PRED_NUM = batch.agent_fut.shape[1]
    FEATURE_NUM = batch.agent_fut.shape[2]

    time = datetime.now().strftime("%b%d_%H-%M-%S")
    writer = SummaryWriter(log_dir=os.path.join(LOGS_DIR, f"./logs/test_data/{time}/"))
    rnn_model = LSTM_Model(
        PRED_NUM, FEATURE_NUM, writer, extracter=lambda x: (x.agent_hist, x.agent_fut)
    )
    # # early_stop_callback = EarlyStopping(
    # #     monitor="val_loss",
    # #     min_delta=0.0,
    # #     patience=1,
    # #     verbose=True,
    # #     mode="min"
    # # )

    trainer = Trainer(
        gpus=0,
        # checkpoint_callback=False,
        accumulate_grad_batches=1,
        max_epochs=50,
        # progress_bar_refresh_rate=10,
    )  # callbacks=[early_stop_callback])

    trainer.fit(rnn_model, train_dataloader, val_dataloader)
    trainer.test(rnn_model, test_dataloader)

    writer.close()

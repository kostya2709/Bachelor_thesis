# import os
# from torch.utils.data import DataLoader
# from trajdata import AgentBatch, UnifiedDataset

from sklearn.model_selection import train_test_split
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler

from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error

# # See below for a list of already-supported datasets and splits.
# dataset = UnifiedDataset(
#     desired_data=["nusc_mini"],
#     data_dirs={  # Remember to change this to match your filesystem!
#         "nusc_mini": "~/datasets/nuscenes"
#     },
# )

# dataloader = DataLoader(
#     dataset,
#     batch_size=64,
#     shuffle=True,
#     collate_fn=dataset.get_collate_fn(),
#     num_workers=os.cpu_count(), # This can be set to 0 for single-threaded loading, if desired.
# )

# batch: AgentBatch
# for batch in dataloader:
#     # Train/evaluate/etc.
#     pass

def read_data( filename: str):
    data = []
    with open( filename) as f:
        for line in f:
            sample = list(map(float, line.split(" ")))
            data.append( sample)
    return data


class TrajDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len( self.data)
    
    def __getitem__(self, index):
        return torch.tensor(self.data[index]), torch.tensor(self.targets[index])


class TrajAccuracy:
    def __init__(self):
        self.value_all = []
        self.value_end = []

    def update( self, pred, ground_truth):
        self.value_all.append( mean_squared_error( pred, ground_truth))
        self.value_end.append( mean_squared_error( pred[-1], ground_truth[-1]))
    
    def __str__(self):
        return str( "All: {}, end: {}", self.value_all, self.value_end)
    
    def compute( self):
        return str( "All: {}, end: {}", np.mean(self.value_all), np.mean(self.value_end))


class LSTM_Model(LightningModule):
    def __init__(self, pred_num: int, embedding_dim=32):
        super().__init__()
        
        self.loss = nn.MSELoss()
        self.valid_accuracy = TrajAccuracy()
        self.test_accuracy = TrajAccuracy()

        self.embeddings_layer = nn.Linear( 1, embedding_dim)
        self.lstm_layer = nn.LSTM(embedding_dim, embedding_dim, batch_first=True)
        self.dropout_layer = nn.Dropout(0.2)
        self.out_layer = nn.Linear(2 * embedding_dim, pred_num)
    
    def forward(self, inputs, labels):
        batch_size = inputs.size(0)
        projections = self.embeddings_layer(inputs)

        # batch_size x seq_len x embedding_dim
        output, (final_hidden_state, final_cell_state) = self.lstm_layer(projections)

        # 2 x batch_size x embedding_dim
        final_hidden_state = final_hidden_state.transpose(0, 1)

        # batch_size x 2 x embedding_dim
        final_hidden_state = final_hidden_state.reshape(batch_size, -1)

        # batch_size x 2*embedding_dim
        hidden = self.dropout_layer(final_hidden_state)
        results = self.out_layer(hidden)
        loss = self.loss(results, labels.float())
        return loss, results
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return [optimizer]
    
    def training_step(self, batch, _):
        inputs, labels = batch
        loss, results = self(inputs, labels)
        print("Training loss:", loss)
        return loss
    
    def validation_step(self, batch, _):
        inputs, labels = batch
        val_loss, results = self(inputs, labels)
        self.valid_accuracy.update(results, labels)
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_acc", self.valid_accuracy)

    def validation_epoch_end(self, outs):
        self.log("val_acc_epoch", self.valid_accuracy.compute(), prog_bar=True)

    def test_step(self, batch, _):
        inputs, labels = batch
        test_loss, results = self(inputs, labels)
        self.test_accuracy.update(results, labels)
        self.log("test_loss", test_loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy)

    def test_epoch_end(self, outs):
        self.log("test_acc_epoch", self.test_accuracy.compute(), prog_bar=True)


if __name__ == "__main__":

    data = read_data( "data")
    target = read_data( "pred")
    train_data, test_data, train_label, test_label = train_test_split( data, target, test_size=0.2, random_state=1)
    train_data, val_data, train_label, val_label = train_test_split( train_data, train_label, test_size=0.2, random_state=1)

    BATCH_SIZE = 32
    PRED_NUM = len(target[0])
    train_data = TrajDataset( train_data, train_label)
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=train_sampler)

    val_data = TrajDataset( val_data, val_label)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    test_data = TrajDataset( test_data, test_label)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    rnn_model = LSTM_Model( PRED_NUM)
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.0,
        patience=1,
        verbose=True,
        mode="min" 
    )

    trainer = Trainer(
        gpus=1,
        #checkpoint_callback=False,
        accumulate_grad_batches=1,
        max_epochs=10,
        #progress_bar_refresh_rate=10,
        callbacks=[early_stop_callback])
    trainer.fit(rnn_model, train_loader, val_loader)
    trainer.test(rnn_model, test_loader)

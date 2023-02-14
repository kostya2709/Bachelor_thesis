import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.utils.tensorboard import SummaryWriter


class TrajAccuracy:
    def __init__(self):
        self.value_all = []
        self.value_end = []

    def update(self, pred, ground_truth):
        diff = torch.linalg.norm(pred - ground_truth, dim=-1)
        self.value_all.append(torch.mean(diff).item())
        self.value_end.append(torch.mean(diff[:, -1]).item())

    def __str__(self):
        return f"All: {self.value_all[-1]}, end: {self.value_end[-1]}"

    def compute(self):
        return f"All: {np.mean(self.value_all)}, end: {np.mean(self.value_end)}"


class LSTM_Model(LightningModule):
    def __init__(
        self,
        pred_num: int,
        feature_num: int,
        writer: SummaryWriter,
        embedding_dim=32,
        extracter=lambda x: x,
    ):
        super().__init__()

        self.loss = nn.MSELoss()
        self.valid_accuracy = TrajAccuracy()
        self.test_accuracy = TrajAccuracy()
        self.writer = writer
        self.extracter = extracter

        self.embeddings_layer = nn.Linear(feature_num, embedding_dim)
        self.lstm_layer = nn.LSTM(
            embedding_dim, embedding_dim, bidirectional=True, batch_first=True
        )
        self.dropout_layer = nn.Dropout(0.2)
        self.out_layer = nn.Linear(2 * embedding_dim, pred_num)

    def forward(self, inputs, labels):
        # inputs: batch_size * seq_len * state_dim
        batch_size = inputs.size(0)
        projections = self.embeddings_layer(
            inputs
        )  # batch_size * seq_len * embedding_dim

        # batch_size x seq_len x embedding_dim
        output, (final_hidden_state, final_cell_state) = self.lstm_layer(projections)

        # 2 x batch_size x embedding_dim
        final_hidden_state = final_hidden_state.transpose(0, 1)

        # batch_size x 2 x embedding_dim
        final_hidden_state = final_hidden_state.reshape(batch_size, -1)

        # batch_size x 2*embedding_dim
        hidden = self.dropout_layer(final_hidden_state)
        results = self.out_layer(hidden)[..., None]
        loss = self.loss(results, labels.float())
        return loss, results

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return [optimizer]

    def training_step(self, batch, _):
        inputs, labels = self.extracter(batch)
        loss, results = self(inputs, labels)
        print("Training loss:", loss)
        self.writer.add_scalar("Loss/train", loss, self.current_epoch)
        return loss

    def validation_step(self, batch, _):
        inputs, labels = self.extracter(batch)
        val_loss, results = self(inputs, labels)
        self.valid_accuracy.update(results, labels)
        self.writer.add_scalar("Loss/val", val_loss, self.current_epoch)
        # self.log("val_loss", val_loss, prog_bar=True)
        # self.log("val_acc", self.valid_accuracy)

    def validation_epoch_end(self, outs):
        self.writer.flush()
        print("Validation epoch end:", self.valid_accuracy.compute())
        # self.log("val_acc_epoch", self.valid_accuracy.compute(), prog_bar=True)

    def test_step(self, batch, _):
        inputs, labels = self.extracter(batch)
        test_loss, results = self(inputs, labels)
        self.test_accuracy.update(results, labels)
        self.writer.add_scalar("Loss/test", test_loss, self.current_epoch)
        # self.log("test_loss", test_loss, prog_bar=True)
        # self.log("test_acc", self.test_accuracy)

    def test_epoch_end(self, outs):
        self.writer.flush()
        print("Test epoch end:", self.valid_accuracy.compute())
        # self.log("test_acc_epoch", self.test_accuracy.compute(), prog_bar=True)

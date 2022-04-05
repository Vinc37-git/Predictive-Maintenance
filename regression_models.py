# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 14:30:42 2021

@author: Fabian
"""

from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from skorch.callbacks import Callback

import torch
import torch.nn as nn
import torch.nn.functional as F


from data_preprocessing import subdivide_dataframe_by_feature


def get_rul_score(gt, pred, return_array=False):
    h = pred - gt
    s = np.zeros(h.shape)
    s[h < 0] = np.exp(-h[h < 0] / 13) - 1
    s[h >= 0] = np.exp(h[h >= 0] / 10) - 1
    if return_array:
        return s
    else:
        return np.sum(s)


class HistoryPlotter(Callback):
    def __init__(self, per_epoch):
        self.per_epoch = per_epoch

    def on_epoch_end(self, net, **kwargs):
        if self.per_epoch:
            clear_output(wait=True)
            fig = plt.figure(figsize=(10, 5))
            fig.suptitle("Training- and Validation-Loss", fontsize=20)
            axis = plt.subplot()
            axis.set_xlabel("Epoch", fontsize=20)
            axis.set_ylabel("Loss", fontsize=20)
            axis.plot(net.history[:, "epoch"],
                      net.history[:, "train_loss"],
                      color="blue", label="training loss", lw=3)
            axis.plot(net.history[:, "epoch"],
                      net.history[:, "valid_loss"],
                      color="orange", label="validation loss", lw=3)
            axis.legend(loc='upper right', fontsize=10)
            axis.grid(True)
            plt.show()
            print("Epoch Duration: {:.2f} s, Total Duration {:.2f} min".format(net.history[-1, "dur"],
                                                                               np.sum(net.history[:, "dur"]) / 60))

    def on_train_end(self, net, **kwargs):
        if not self.per_epoch:
            fig = plt.figure(figsize=(10, 5))
            fig.suptitle("Training- and Validation-Loss", fontsize=20)
            axis = plt.subplot()
            axis.set_xlabel("Epoch", fontsize=20)
            axis.set_ylabel("Loss", fontsize=20)
            axis.plot(net.history[:, "epoch"],
                      net.history[:, "train_loss"],
                      color="blue", label="training loss", lw=3)
            axis.plot(net.history[:, "epoch"],
                      net.history[:, "valid_loss"],
                      color="orange", label="validation loss", lw=3)
            axis.legend(loc='upper right', fontsize=10)
            axis.grid(True)
            plt.show()
            print("Total Duration {:.2f} min".format(np.sum(net.history[:, "dur"]) / 60))


def evaluate_regression_model(pred, gt,
                              all_units_test,
                              all_time_in_cycles_test,
                              target_def="min",
                              choice="widest"):

    # Calculate and print Scores
    R2_score = r2_score(gt, pred)
    rul_score = get_rul_score(gt, pred)
    print("R2-Score:\t", R2_score)
    print("RUL-Score:\t", rul_score)

    # print distribution of prediction
    errors = np.abs(pred - gt)
    fig, ax = plt.subplots(1, 3, figsize=(25, 5))
    ax[0].hist(gt, bins=100)
    ax[0].set_title("GT targets")
    ax[1].hist(pred, bins=100)
    ax[1].grid(True)
    #ax[1].hist(gt , bins=100, alpha=0.6)
    ax[1].set_title("Predicted targets")
    ax[2].hist(errors, bins=100)
    ax[2].set_title("Errors")
    ax[2].grid(True)

    # Einen noch besseren Eindruck über die Qualität des Modells liefert uns die Analyse einzelnen Versuchsläufe.
    # Hier plotten wir die tatsächliche sowie die vorhergesagte Restzeit über den Versuchsverlauf.
    result_df = pd.DataFrame(data={"preds": pred.flatten(),
                                   "labels": gt.flatten(),
                                   "errors": errors.flatten()})

    # Add Unit number and time_in_cycles to the dataframe:
    result_df["unit_number"] = all_units_test
    if target_def == "min":
      result_df["time_in_cycles"] = all_time_in_cycles_test.max(axis=1)
 

    list_of_result_df = subdivide_dataframe_by_feature(result_df,
                                                       feature="unit_number",
                                                       drop_feature=True)

    # plot either for 8 random units predicted RUL over Time Cycles
    # or for 8 units with the widest "time_in_cycles"-span
    if choice == "random":
        l = np.random.randint(low=0, high=len(list_of_result_df), size=(8,))
    elif choice == "widest":
        diff = []
        for df in list_of_result_df:
            diff.append(df["time_in_cycles"].max() - df["time_in_cycles"].min())
        l = np.argsort(diff)[-8:]

    fig, ax = plt.subplots(2, 4, figsize=(25, 12))
    ax = ax.flatten()
    fig.suptitle("Predictions per Unit", fontsize=20)

    for i, k in enumerate(list(l)):
        df = list_of_result_df[k].sort_values("time_in_cycles")
        ax[i].plot(df["time_in_cycles"].values,
                   df["labels"].values, label="actual RUL")
        ax[i].plot(df["time_in_cycles"].values,
                   df["preds"].values, label="Predicted RUL", c="r", marker='*')
        ax[i].set_ylabel("RUL", fontsize=10)
        ax[i].set_xlabel("Time Cycle", fontsize=10)
        ax[i].set_title("Unit {}".format(int(k)), fontsize=15)
        ax[i].grid(b=True)
        ax[i].legend()


class ConvRegressor(nn.Module):

    def __init__(self, start_filters: int, channels: int, seq_length):
        super().__init__()
        self._start_filters = start_filters
        self._channels = channels
        self._seq_length = seq_length
        self.conv1 = nn.Conv1d(channels, start_filters * 2, 3, padding=1)
        self.conv2 = nn.Conv1d(start_filters * 2, start_filters * (2 ** 2), 3, padding=1)
        self.conv3 = nn.Conv1d(start_filters * (2 ** 2), start_filters * (2 ** 3), 3, padding=1)
        self.conv4 = nn.Conv1d(start_filters * (2 ** 3), start_filters * (2 ** 4), 3, padding=1)
        self.conv5 = nn.Conv1d(start_filters * (2 ** 4), start_filters * (2 ** 5), 3, padding=1)

        self.lin1 = nn.Linear(self._start_filters * (2 ** 5) * int(self._seq_length / (2 ** 5)), 128)
        self.lin2 = nn.Linear(128, 1)

        self.pool = nn.MaxPool1d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))

        prev_size = x.size()
        x = x.view(-1, self._start_filters * (2 ** 3) * int(self._seq_length / (2 ** 3)))
        x = F.dropout(F.relu(self.lin1(x)), p=0.2)
        x = self.lin2(x)
        return x


class DNNRegressor(nn.Module):
    def __init__(self,
                 input_size,
                 n_layer=3,
                 nh=[10, 1, 1],
                 hactfn=[torch.sigmoid, torch.sigmoid, None],
                 p_dropout=0,
                 output_size=1,
                 oactfn=F.relu,
                 device="cpu"):

        '''
        input_size  : input size as tuple of (features, seq_length)
        n_layer     : number of layers
        nh          : list of hidden neurons per layer
        hactfn      : lsit of hidden activation functions per layer
        p_dropout   : list of dropout probabailities per layer
        output_size : output neurons
        oactfn      : output activation function
        '''

        super().__init__()
        self.features = input_size[0]
        self.seq_length = input_size[1]
        self.ni = input_size[0] * input_size[1]   # input neurons
        self.n_layer = n_layer
        self.nh = nh
        self.hactfn = hactfn
        self.p_dropout = p_dropout
        if p_dropout is not None:
            self.do = nn.Dropout(p=p_dropout)
        self.no = output_size
        self.oactfn = oactfn
        self.device = device

        # Check length of lists. if they are given as int, make lists
        if n_layer != 1 and type(nh) != list and type(hactfn) != list:
            self.nh = [nh for _ in range(n_layer)]
            self.hactfn = [hactfn for _ in range(n_layer)]
        else:
            assert n_layer >= 1,       "n_layer >= 1 required"
            assert len(nh) == n_layer, "Length of nh list must match the number of layers"
            assert len(hactfn) == n_layer, "Length of hactfn list must match the number of layers"

        # instanciate list of linear superposition cells
        self.lins = [nn.Linear(self.ni, self.nh[0])] +\
            [nn.Linear(self.nh[layer-1], self.nh[layer]) for layer in range(1, self.n_layer)]

        # send to GPU if required
        self.lins = nn.ModuleList(self.lins)

        # output cell
        self.linout = nn.Linear(self.nh[-1], self.no)

        # old
        '''
        input_dim = input_size[0] * input_size[1]   # input neurons
        self.lin1 = nn.Linear(    input_dim, 10 * input_dim)
        self.lin2 = nn.Linear(10 * input_dim,     input_dim)
        self.lin3 = nn.Linear(    input_dim, input_dim)
        self.lin4 = nn.Linear(    input_dim, 1)
        '''

    def forward(self, x):
        # input has shape: (batch_size, features, seq_length) -> flatten 2nd and 3rd dimension
        x = torch.flatten(x, start_dim=1, end_dim=2).to(self.device)

        # hidden layers
        for i in range(len(self.lins)):
            x = self.lins[i](x)
            if self.hactfn[i] is not None:
                x = self.hactfn[i](x)
            if self.p_dropout is not None:
                x = self.do(x)

        # last layer to output
        x = self.linout(x)
        if self.oactfn is not None:
            x = self.oactfn(x)

        # old
        '''
        x = torch.sigmoid(self.lin1(x))
        x = torch.sigmoid(self.lin2(x))
        x = self.lin3(x) #torch.relu(self.lin3(x))
        x = F.relu(self.lin4(x))
        '''

        return x


class RNNRegressor(nn.Module):
    def __init__(self, 
                 input_size, 
                 no=1, 
                 nh=10, 
                 nlayers=1, 
                 hactfn=nn.Tanh(), 
                 actfn=None, 
                 pDropout=0, 
                 rnnType="lstm", 
                 bidir=False, 
                 state="stateless",
                 device="cpu"):

        super().__init__()

        
        '''
        nh should be HIDDEN_FEATURES
        '''

        self.batches    = input_size[0]
        self.features   = input_size[1]
        self.seq_length = input_size[2]

        self.no = no
        self.nh = nh
        self.nlayers = nlayers
        self.rnnType = rnnType

        self.gru  = nn.GRU(self.features, self.nh, self.nlayers, dropout=pDropout, bidirectional=bidir)
        self.lstm = nn.LSTM(self.features, self.nh, self.nlayers, dropout=pDropout, bidirectional=bidir)

        # hidden states
        self.h = None
        self.c = None

        # bidirectional
        self.bidir     = bidir
        self.bidir_int = 1 if bidir is False else 2  

        # linear output layer    
        self.out = nn.Linear(self.bidir_int * self.nh, self.no)

        # hidden and output activation function
        self.hactfn = hactfn 
        self.actfn  = actfn

        self.state = state
        self.device = device
        

    def forward(self, x):
      '''
      First part is about initialising the hidden states. Can be stateless or stateful
      '''

      BATCHES, FEATURES, SEQ_LENGTH = (x.shape)

      # permute to SEQ_LENGTH, BATCHES, FEATURES
      x = x.permute(2, 0, 1)

      # hidden state initialization

      if self.rnnType == "gru":
        if self.h is None or self.state == "stateless":
          # initial hidden state for each element in batch
          self.h = h = torch.zeros(self.bidir_int * self.nlayers, BATCHES, self.nh, device=self.device)
        else:
          # stateful: the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch
          h = self.h
          # in case a batch is not complete, crop the hidden state but dont save it
          if BATCHES is not self.h.shape[1]:
             h = torch.split(self.h, BATCHES, dim=1)[0].contiguous()

      elif self.rnnType == "lstm":
        if self.h is None or self.state == "stateless":
          # initial hidden state for each element in batch
          self.h = h = torch.zeros(self.bidir_int * self.nlayers, BATCHES, self.nh, device=self.device)
          self.c = c = torch.zeros(self.bidir_int * self.nlayers, BATCHES, self.nh, device=self.device)
        else:
          # stateful: the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch
          (h, c) = (self.h, self.c)
          # in case a batch is not complete, crop the hidden state but dont save it
          if BATCHES is not self.h.shape[1]:
            h = torch.split(self.h, BATCHES, dim=1)[0].contiguous()
            c = torch.split(self.c, BATCHES, dim=1)[0].contiguous()

      '''
      ### OLD but explains the algorithm well !! Thats why i keep it
      # RNN cells stacked for nlayers
      for l in range(self.nlayers):
        seq_out = []
        #print(l, hs.shape, h.shape)
        for i in range(SEQ_LENGTH):
          if self.rnnType == "gru":
            h  = self.grus[l](hs[i], h)
          else:
            h, c = self.lstms[l](hs[i], (h, c))
            hs = self.do(h)
          seq_out.append(h)
        hs = (seq_out)
        h = self.hactfn(h)
        h = self.do(h)
      ''' 

      if self.rnnType == "gru":
        seq_out, h = self.gru(x, h)
        if BATCHES is self.h.shape[1]:  # save hidden states if batch is complete
          self.h = h.detach()

      elif self.rnnType == "lstm":
        seq_out, (h, c) = self.lstm(x, (h, c))
        if BATCHES is self.h.shape[1]:  # save hidden states if batch is complete
          self.h = h.detach()
          self.c = c.detach()

      y = seq_out[-1]

      if self.hactfn is not None:
        y = self.hactfn(y)

      ### Dense Layer
      if self.actfn is None:
        y = self.out(y)
      else:
        y = self.actfn(self.out(y))

      return y #, (h, c)


if __name__ == "__main__":
    pass

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
  """Long Short-Term memory network."""

  def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
    super(LSTMModel, self).__init__()
    # Hidden dimensions
    self.hidden_dim = hidden_dim

    # Number of hidden layers
    self.layer_dim = layer_dim

    # Building your LSTM
    # batch_first=True causes input/output tensors to be of shape
    # (batch_dim, seq_dim, feature_dim)
    self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

    # Readout layer
    self.fc = nn.Linear(hidden_dim, output_dim)

  def forward(self, x, x_lengths):
    device = x.device

    # Initialize hidden state with zeros
    h0 = torch.zeros(self.layer_dim, x.size(0),
                     self.hidden_dim).requires_grad_()
    h0 = h0.to(device)

    # Initialize cell state
    c0 = torch.zeros(self.layer_dim, x.size(0),
                     self.hidden_dim).requires_grad_()
    c0 = c0.to(device)

    # pack_padded_sequence so that padded items in the sequence won't be shown
    # to the LSTM.
    x = torch.nn.utils.rnn.pack_padded_sequence(x,
                                                x_lengths,
                                                enforce_sorted=False,
                                                batch_first=True)

    # We need to detach as we are doing truncated backpropagation through time
    # (BPTT).
    # If we don't, we'll backprop all the way to the start even after going
    # through another batch.
    out, (hn, _) = self.lstm(x, (h0.detach(), c0.detach()))

    # undo the packing operation
    out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

    # We just want the last hidden states
    # Apply fc layer to them
    res = self.fc(hn[-1])

    return res

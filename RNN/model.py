import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

class RNN(nn.Module):
    def __init__(self, rnn_type: str, input_size: int, hidden_size: int, num_layers: int = 1,
                batch_size: int = 1, dropout: float = 0.5, embedding_dim: int = 64):
        """
        rnn_type: str -> type of RNN (RNN/LSTM)
        input_size: int -> # of features in x
        hidden_size: int -> # of features in hidden layer h
        num_layers: int = 1 -> # of recurrent layers
        batch_size: int = 1 -> size of input batch
        droput: float = 0.5 -> p value of dropout
        embedding_dim: int -> word embedding dimensions
        """
        super(RNN, self).__init__()
        
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim

        self.drop = nn.Dropout(dropout)
        
        self.embeddings = self.word_embedding(self.input_size, self.embedding_dim) # vocab_size, embedding_dim
        
        # if True: [input, output] -> (batch, seq_len, input_size), else -> (seq_len, batch, input_size)
        batch_first = False
        
        if self.rnn_type in ["RNN", "LSTM"]:
            # input_size is going to be # of embedding dimensions we've converted to
            self.rnn = getattr(nn, self.rnn_type)(input_size = self.embedding_dim, hidden_size = self.hidden_size,
                                            batch_first = batch_first, num_layers = self.num_layers) # only use dropout if `num_layers` > 1
        else:
            raise ValueError("not a valid model choice")
            
        self.logits = nn.Linear(hidden_size, self.input_size).apply(utils.glorot_normal_initializer)
            
    def forward(self, x, hidden):
        x_e = self.embeddings(x)
        
        if self.num_layers > 1:
            x_e = self.drop(x_e)
            
        x_e = x_e.view(x_e.shape[1], 1, self.embedding_dim) # (seq_len, batch, input_size)
        #x_e = x_e.view(1, x_e.shape[1], self.embedding_dim) # (batch, seq_len, input_size)
        
        out, h = self.rnn(x_e, hidden)
        logits = self.logits(out.view(-1, self.hidden_size))
        probs = F.log_softmax(logits, dim = 1)

        return probs, h
        
    def init_hidden(self):
        if self.rnn_type == "LSTM":
            H = (torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
                        torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
        else:
            # (num_layers * num_directions, batch, hidden_size)
            H = torch.autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)) # batch_first = False
            #H = torch.autograd.Variable(torch.zeros(self.num_layers, 16, self.hidden_size))

        return H
    
    def word_embedding(self, vocab_size: int, embedding_dim: int):
        return nn.Embedding(vocab_size, embedding_dim)

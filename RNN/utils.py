import os
import logging

import tqdm

import numpy as np
import torch
import torch.nn as nn

def glorot_normal_initializer(m):
    """ Applies Glorot Normal initialization to layer parameters.
    
    "Understanding the difficulty of training deep feedforward neural networks" 
    by Glorot, X. & Bengio, Y. (2010)

    Args:
        m (nn.Module): a particular layer whose params are to be initialized.
    """
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)

def tokenize(levels: list) -> tuple:
    #idx_to_col = list(set([col for level in levels for col in level]))
    #idx_to_col.append("EOS")

    _ = set([col for level in levels for col in level])
    idx_to_col = {i: col for i, col in enumerate(_)}
    idx_to_col.update({len(idx_to_col): "EOS"})

    #col_to_idx = {col: idx_to_col.index(col) for col in idx_to_col}
    col_to_idx = {col: i for level in levels for i, col in enumerate(level)}
    col_to_idx.update({len(col_to_idx): "EOS"})

    assert len(idx_to_col) ==  len(col_to_idx), "something went wrong when tokenizing"

    return (idx_to_col, col_to_idx) 

# slightly too convoluted
def prepare_sequence_input(seq: np.ndarray, to_ix: dict()):
    if seq.shape[-1] == 1:
        seq = [col[0] for col in seq]
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype = torch.long)

def prepare_sequence_target(seq: np.ndarray, to_ix: dict()):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype = torch.long)

class SMB1:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def _transpose_level(self, level: list) -> list:
        new_level = list(zip(*level))
        new_level = [''.join(new_level[i]) for i in range(len(new_level))]
        return new_level
            
    def output_level(self, level: list, name: str):
        level = self._transpose_level(level)
        
        if not os.path.exists(self.data_dir + "gen_levels/"):
            os.makedirs(self.data_dir + "gen_levels/")

        with open(f"{self.data_dir}gen_levels/{name}", "w") as f:
            for col in zip(level):
                f.write('{}\n'.format(*col))
    
class SMM2:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    # probably better ways to do this...
    def _transpose_level(self, level: list) -> list:
        new_level = list(zip(*level))[::-1]
        ["".join(new_level[i]) for i in range(len(new_level))]
        return new_level

    def output_level(self, level: list, name: str):
        level = self._transpose_level(level)
        
        if not os.path.exists(self.data_dir + "gen_levels/"):
            os.makedirs(self.data_dir + "gen_levels/")

        with open(f"{self.data_dir}gen_levels/{name}", "w") as f:
            for col in level:
                f.write("".join(str(i) for i in col) + "\n")

class TQDMLoggingHandler(logging.Handler):
    def __init__(self, level = logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)  

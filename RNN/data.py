import os
import typing

import numpy as np

# return list of file names
def get_level_files(data_dir: str) -> list:
    level_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    return level_files

def gen_data(levels: list, step: int, max_length: int):
    # step: window size in level sequence
    # max_length: max sequence length (seq_len)
    l = np.concatenate(levels)
    
    sequences = list() # sequences as input
    next_seq = list() # target sequence from sequence[t] + step

    for i in range(0, len(l) - max_length, step):
        sequences.append(l[i:i + max_length])
        next_seq.append(l[i + max_length])

    del l
    
    return sequences, next_seq

def one_hot(sequences: list, max_length: int) -> typing.Tuple[np.ndarray, np.ndarray]:
    X = np.zeros((len(sequences), max_length, len(idx_to_col)), dtype = np.bool)
    y = np.zeros((len(sequences), len(idx_to_col)), dtype = np.bool)

    for i, sequence in enumerate(sequences):
        for t, col in enumerate(sequence):
            X[i, t, col_to_idx[col]] = 1
        y[i, col_to_idx[next_seq[i]]] = 1
        
    assert X.shape[0] == y.shape[0], "somethings gone wrong!"
    assert X.shape[-1] == y.shape[1], "somethings gone wrong!"
        
    return (X, y)

#X, y = one_hot(sequences, max_length)

def train_test_split(data):
    data = np.concatenate(data)
    split_index = int(0.8 * len(data))
    train, test = data[:split_index], data[split_index:]
    
    train, test = np.array(train), np.array(test)
    
    return train, test

def batchify(data, bsz):
    nbatch = data.shape[0] // bsz
    data = data[:(nbatch * bsz)]
    data = data.reshape(-1, bsz)
    return data

def get_batch(source, seq_len: int, idx: int):
    seq = min(seq_len, len(source) - 1 - idx)
    data = source[idx:idx + seq]
    target = source[idx + 1:idx + 1 + seq].reshape(-1)
    return data, target

#train_data, test_data = train_test_split(levels)
#train_data = batchify(train_data, 1)
#test_data = batchify(test_data, 1)

##########################################################################################
##########################################################################################

class MW:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.level_files = get_level_files(self.data_dir)

    # process levels into column-wise representation
    def init_process_levels(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        for lf in self.level_files:
            tmp = []
            with open(self.data_dir + lf, "r") as f:
                for line in f:
                    tmp.append(line.rstrip())

            # "transpose" the level
            tmp = zip(*tmp)
            level = []

            for col in list(tmp):
                level.append("".join(col))   

            with open(f"{self.data_dir}col_levels/{lf}", "w") as f:
                for col in level:
                    f.write(f"{col}\n")

        self.data_dir += "col_levels"

    # read in level data
    def read_level(self, level_name: list) -> list:
        levels = list()
        
        for i in level_name:
            level = list()
            
            with open(f"{self.data_dir}/{i}") as f:
                for line in f:
                    level.append(line.rstrip())
                    
            levels.append(level)
                
        return levels

#level = read_level(self.data_dir + "col_levels/", "mario-1-1.txt")
#level.extend(read_level(self.data_dir + "col_levels/", "mario-1-2.txt"))

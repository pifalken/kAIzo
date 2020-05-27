"""
WARNING: this is ugly. it was converted from a iPython notebook
"""

import os
import logging
import random
from collections import Counter
# from typing import Generator

import numpy as np

"""
# return generator with file names
def get_level_files(data_dir: str) -> Generator:
    level_files = (f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f)))
    return level_files
"""

# return list of file names
def get_level_files(data_dir: str) -> list:
    level_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    return level_files

def init_process_levels(data_dir: str):
    if not os.path.exists(data_dir + "col_levels/"):
        os.makedirs(data_dir + "col_levels/")

    level_files = get_level_files(data_dir)
    for lf in level_files:
        tmp = []
        with open(data_dir + lf, "r") as f:
            for line in f:
                tmp.append(line.rstrip())

        # "transpose" the level
        tmp = zip(*tmp)
        level = []

        for col in list(tmp):
            level.append("".join(col))   

        with open(f"{data_dir}col_levels/{lf}", "w") as f:
            for col in level:
                f.write(f"{col}\n")
                
data_dir = "../data/smb/Processed/"
init_process_levels(data_dir)

def read_level(data_dir: str, level_name: str) -> list:
    level = []
    
    with open(data_dir + level_name) as f:
        for line in f:
            level.append(line.rstrip())
            
    return level

level = read_level(data_dir + "col_levels/", "mario-1-1.txt")
level.extend(read_level(data_dir + "col_levels/", "mario-1-2.txt"))
level.extend(read_level(data_dir + "col_levels/", "mario-1-3.txt"))

#data_dir = "../data/smm2/"
#level = read_level(data_dir, "course_data_000.txt")
#level.extend(read_level(data_dir, "course_data_003.txt"))
#level.extend(read_level(data_dir, "course_data_004.txt"))
#level.extend(read_level(data_dir, "course_data_005.txt"))
#level.extend(read_level(data_dir, "course_data_006.txt"))
#level = read_level(data_dir, "course_data_006.txt")

# general ngram generator
freq = dict()

for i in range(len(level) - 1):
    ngram = (level[i], level[i + 1])
    
    if freq.get(ngram):
        freq[ngram] += 1
    else:
        freq[ngram] = 1
        
# divide freqs by total items
for k, v in freq.items():
    freq[k] = v / len(freq)
    
# normalize between 0 - 1
#normed = [float(v) / sum(freq.values()) for v in freq.values()]
_ = sum(freq.values())
freq_norm = {k: (float(v) / _) for k, v in freq.items()} # this is a bit ugly and dangerous but...

# could also use random.choices() with the freq_norm values
def weighted_rnd(choices: list) -> str:
    total = sum(v for k, v in choices)
    r = random.uniform(0, total)
    
    upto = 0
    for k, v in choices:
        upto += v
        if upto > r:
            return k

def generate_level(curr_col: str, ngrams: dict, n: int = 10):
    print(curr_col)
    new_level = list()

    for i in range(n):
        choices = [(k, v) for k, v in ngrams.items() if curr_col in k]
        if not choices:
            break
            
        curr_col = weighted_rnd(choices)[1]
        new_level.append(curr_col)
        
    return new_level

def _transpose_level_smb(level: list) -> list:
    print(len(level))
    new_level = list(zip(*level))
    new_level = [''.join(new_level[i]) for i in range(len(new_level))]
    return new_level

# probably better ways to do this...
def _transpose_level_smm2(level: list) -> list:
    new_level = list(zip(*level))[::-1]
    ["".join(new_level[i]) for i in range(len(new_level))]
    return new_level
        
def _output_level_smb(level: list, name: str):
    level = _transpose_level_smb(level)
    
    if not os.path.exists(data_dir + "gen_levels/"):
        os.makedirs(data_dir + "gen_levels/")

    with open(f"{data_dir}gen_levels/{name}", "w") as f:
        for col in zip(level):
            f.write('{}\n'.format(*col))
            
def _output_level_smm2(level: list, name: str):
    level = _transpose_level_smm2(level)
    
    if not os.path.exists(data_dir + "gen_levels/"):
        os.makedirs(data_dir + "gen_levels/")

    with open(f"{data_dir}gen_levels/{name}", "w") as f:
        for col in level:
            f.write("".join(str(i) for i in col) + "\n")
    
#start = '---------XXXXX' # smb1
start = "XB--------XE--XXXX----------" # smm2
new_level = generate_level(start, freq, 20)

#_output_level_smb(new_level, "mario-1-1_NEW.txt") # smb1
_output_level_smm2(new_level, "mario-1-1_NEW.txt") # smm2

# technically we can also generate n-grams using:
# return list(zip(*[tokens[i:] for i in range(n)]))
def gen_ngram(level: list, n: int):
    ngrams = dict()
    
    for i in range(len(level) - n):
        gram = tuple(level[i:i + n])
        
        if ngrams.get(gram):
            ngrams[gram] += 1
        else:
            ngrams[gram] = 1
        
    for k, v in ngrams.items():
        ngrams[k] = v / len(ngrams)
        
    _ = sum(ngrams.values())
    ngrams = {k: (float(v) / _) for k, v in ngrams.items()}
    
    return ngrams

three_gram = gen_ngram(level, 3)
four_gram = gen_ngram(level, 4)

#start_points = ['---------XXXXX', '---------Q---X', '--------------', '----------<[[X'] # smb1
start_points = ["X------XXXXXXXXXXXXXXXXXXXX-", "XB--------XE--XXXX----------"] #smm2
for i in range(len(start_points)):
    new_level = generate_level(start_points[i], three_gram, 160)
    
    _ = f'mario-1-1_{i}.txt'
    #_output_level_smb(new_level, _) # smb1
    _output_level_smm2(new_level, _) # smm2

class MarkovChain():
    def __init__(self):
        self.markov_chain = dict()
        
    def _transition(self, key, value: str):
        """
        internal function, generates markov chain transitions
        """
        if key not in self.markov_chain:
            self.markov_chain[key] = []
            
        self.markov_chain[key].append(value)

    def chain(self, tokens: list, n: int):
        """
        tokens: list of all possible states in the markov chain (i.e, unique cols in the level)
        n: markov chain order
        """
        ngrams = gen_ngram(tokens, n)
                        
        for ngram in ngrams:
            self._transition(ngram[:2], ngram[2])
            
    def generate_level(self, curr_col: tuple, n: int = 10) -> list:
        """
        curr_col: tuple of starting level columns
        n: how many new cols to generate
        """
        new_level = list()
        new_level.extend(curr_col)
        
        for i in range(n):
            next_col = self.markov_chain.get(curr_col)
            next_col = random.sample(next_col, 1)[0]
            
            new_level.append(next_col)            
            curr_col = (curr_col[1], next_col)
            
        return new_level

m = MarkovChain()
m.chain(level, 3)
m_level = m.generate_level((('-------------X', '-------------X')), 100) # smb1
#m.generate_level((("X------XXXXXXXXXXXXXXXXXXXX-", "X------XXXXXXXXXXXXXXXXXXXX-")), 50) # smm2

m_level
_output_level_smb(m_level, "mAIrio.txt")

import os
import sys

import numpy as np
import torch

import data
import utils

# @TODO: add seed for repro

def generate(weights, game: str, diversity: float, idx_to_col: dict, start_slice: str,
                level_length: int, device):
    with open(weights, "rb") as f:
        rnn = torch.load(f, map_location = device)

    rnn.eval()

    with torch.no_grad():
        hidden = rnn.init_hidden()
        if device.type == "cuda":
            if type(hidden) == tuple:
                hidden = tuple([w.to(device) for w in hidden])
            else:
                hidden = hidden.to(device)

        level_slice = torch.tensor(utils.prepare_sequence_target(np.array([start_slice]), col_to_idx)).unsqueeze(0) 

        new_level = list()
        new_level.append(int(level_slice[0]))

        for i in range(level_length):
            output, h = rnn(level_slice.to(device), hidden)

            probs = output.exp().detach()
            slices = torch.multinomial(torch.tensor(probs), 1)[0]

            new_level.append(slices[0])
            level_slice.fill_(slices[0])

    generated_level = [idx_to_col[int(idx)] for idx in new_level]
    mw = utils.SMB1("data/") if game == "SMB1" else utils.SMM2("data/")
    mw.output_level(generated_level, "sample.txt")

if __name__ == "__main__":
    weights = sys.argv[1]
    game = sys.argv[2]
    data_dir = sys.argv[3]
    diversity = sys.argv[4]
    start = sys.argv[5]
    level_length = sys.argv[6]

    if os.path.isdir(data_dir):
        print(f"processing data from {data_dir}")

        mw = data.MW(data_dir)
        if game == "SMB1":
            mw.init_process_levels()

        levels = mw.read_level([i for i in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, i))])

        idx_to_col, col_to_idx = utils.tokenize(levels)
    else:
        raise Exception(f"directory {data_dir} does not exist!")

    device = torch.device("cuda") if sys.argv[7] == "cuda" else torch.device("cpu")

    generate(weights, str(game), float(diversity), idx_to_col, str(start), int(level_length), device)

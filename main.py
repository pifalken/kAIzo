import os
import time

import argparse
import logging
import coloredlogs
import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import model
import utils
import data

parser = argparse.ArgumentParser(description = "Mario level generator")

# general configs
parser.add_argument("--game", type = str, help = "game style to use [SMB1, SMM2]", default = "SMB1")
parser.add_argument("--data", type = str, help = "location of dataset to use")
parser.add_argument("--save", type = str, help = "name to save final model", default = "mario")
parser.add_argument("--show_me_loss", action = "store_true", help = "show pretty logs after training")
parser.add_argument("--log-interval", type = int, help = "log output iteration value", default = 200)
# model configs
parser.add_argument("--rnn_type", type = str, help = "type of RNN to use [RNN, LSTM]", default = "RNN")
parser.add_argument("--seq_len", type = int, help = "sequence length of input data", default = 16)
parser.add_argument("--step_size", type = int, help = "size of window when generating sequences -> next_seq (target)", default = 1)
parser.add_argument("--h", type = int, help = "size of hidden layer", default = 32)
parser.add_argument("--w", type = int, help = "dimension of word embeddings to create", default = 64)
parser.add_argument("--nl", type = int, help = "number of layers to use", default = 1)
parser.add_argument("--nepochs", type = int, help = "number of epochs", default = 5)
parser.add_argument("--bs", type = int, help = "batch size", default = 1)
parser.add_argument("--batch_first", action = "store_true", help = "`batch_first` parameter for torch RNN model")
parser.add_argument("--lr", type = float, help = "learning rate", default = 0.0005)
parser.add_argument("--sched", action = "store_true", help = "flag to toggle scheduler for LR optimization")
parser.add_argument("--droput", type = float, help = "dropout value", default = 0.5)
parser.add_argument("--cuda", action = "store_true", help = "use CUDA")

args = parser.parse_args()

SAVE_FILE = args.save

def train(train_data, test_data, rnn_type: str, num_epochs: int, input_size: int, hidden_size: int, num_layers: int,
        embedding_dim: int, lr: float, seq_len: int, batch_size: int, col_to_idx: dict, idx_to_col: dict,
        log_interval: int, game: str, device):

    rnn = model.RNN(rnn_type, input_size, hidden_size, num_layers, batch_size, embedding_dim = embedding_dim).to(device)

    if device.type == "cuda":
        rnn = rnn.cuda()

    criterion = nn.NLLLoss(reduction = "mean")
    optimizer = optim.Adam(rnn.parameters(), lr = lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "min", min_lr = 1e-6,
                                                    factor = 0.1, patience = 5, verbose = True)

    epoch_val_loss = list()
    total_loss = 0.0
    best_val_loss = None

    print("-" * 80)
    logging.info("TRAINING:")
    logging.info("{} on {}".format(game, rnn_type))
    logging.info("params -> seq_len: {} \t hidden_size: {} \t embedding_dim: {}".format(seq_len, hidden_size, embedding_dim))
    logging.info("\t\tnum_layers: {} \t nepochs: {} \t batch_size: {} \t lr: {}""".format(num_layers, num_epochs,
            batch_size, lr))
    print("-" * 80)
    time.sleep(2)

    for epoch in range(num_epochs):
        rnn.train()

        epoch_loss = 0.0
        epoch_start_time = time.time()
        for batch, i in tqdm.tqdm(enumerate(range(0, len(train_data) - 1, batch_size)), desc = "epoch[{}/{}]".format(epoch, num_epochs),
                                       leave = False, total = len(train_data) - 1):

            hidden = rnn.init_hidden()
            if device.type == "cuda":
                if type(hidden) == tuple:
                    hidden = tuple([w.to(device) for w in hidden])
                else:
                    hidden = hidden.to(device)

            rnn.zero_grad()
            optimizer.zero_grad()

            level_slice, target = data.get_batch(train_data, seq_len, i)
            level_slice = torch.tensor(utils.prepare_sequence_input(level_slice, col_to_idx)).unsqueeze(0)
            target = torch.tensor(utils.prepare_sequence_target(target, col_to_idx)).to(device)
            
            # output -> log_softmax probabilities
            output, h = rnn(level_slice.to(device), hidden)

            loss = criterion(output, target)
            loss.backward()
            
            # prevent the exploding &|| vanishing gradient problem in RNNs
            # https://github.com/pytorch/examples/blob/984700d30e9bf1f2dc735e8a80de37cfa538ae17/word_language_model/main.py#L178
            torch.nn.utils.clip_grad_norm_(rnn.parameters(), 50.0)
            for p in rnn.parameters():
                p.data.add(-lr, p.grad.data)

            optimizer.step()
            epoch_loss += loss.item()
            total_loss += loss.item()
            
            if batch % log_interval == 0:
                _el = epoch_loss / log_interval
                logging.info("epoch[{}/{}] \t time: {:5.2f}s  \t epoch loss: {:.4f} \t total loss: {:.4f}".format(
                    epoch + 1, num_epochs, (time.time() - epoch_start_time), _el, total_loss / log_interval))
                total_loss = 0.0
                del _el
        
        val_loss = evaluate(rnn, test_data, criterion, batch_size, seq_len, device)
        epoch_val_loss.append(val_loss)
        logging.info(f"eval loss: {val_loss:.4f}")
        scheduler.step(val_loss)

        if not best_val_loss or val_loss < best_val_loss:
            best_val_loss = val_loss
            logging.info(f"saving new model weights: {best_val_loss = }")
            with open(f"{SAVE_FILE}{game}_{rnn_type}.pt", "wb") as f:
                torch.save(rnn, f)

        sample(rnn, col_to_idx, idx_to_col, seq_len, "XX---XXXXXXXXX--------------", game, device)
        #sample(rnn, col_to_idx, idx_to_col, seq_len, "---------XXXXX", game, device)
        
def evaluate(rnn, test_data, criterion, batch_size: int, seq_len: int, device):
    logging.info("evaluating model...")

    rnn.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch, i in tqdm.tqdm(enumerate(range(0, len(test_data) - 1, batch_size))):            
            hidden = rnn.init_hidden()
            if device.type == "cuda":
                if type(hidden) == tuple:
                    hidden = tuple([w.to(device) for w in hidden])
                else:
                    hidden = hidden.to(device)

            level_slice, target = data.get_batch(test_data, seq_len, i)
            level_slice = torch.tensor(utils.prepare_sequence_input(level_slice, col_to_idx)).unsqueeze(0)
            target = torch.tensor(utils.prepare_sequence_target(target, col_to_idx)).to(device)
            
            output, h = rnn(level_slice.to(device), hidden)

            loss = criterion(output, target)

            total_loss += len(level_slice) * loss.item()

    return total_loss / len(test_data)

def sample(rnn, col_to_idx: dict, idx_to_col: dict, seq_len: int, start: str, game: str, device):
    logging.info("generating a mid-sample...")
    rnn.eval()

    with torch.no_grad():
        hidden = rnn.init_hidden()
        if device.type == "cuda":
            if type(hidden) == tuple:
                hidden = tuple([w.to(device) for w in hidden])
            else:
                hidden = hidden.to(device)

        level_slice = torch.tensor(utils.prepare_sequence_target(np.array([start]), col_to_idx)).unsqueeze(0) 

        new_level = list()
        new_level.append(int(level_slice[0]))

        for i in range(seq_len):
            output, h = rnn(level_slice.to(device), hidden)

            probs = list(np.exp(output.cpu().detach().numpy()))
            slices = torch.multinomial(torch.tensor(probs), 1)[0]

            new_level.append(slices[0])
            level_slice.fill_(slices[0])

    generated_level = [idx_to_col[int(idx)] for idx in new_level]
    mw = utils.SMB1("data/") if game == "SMB1" else utils.SMM2("data/")
    mw.output_level(generated_level, "sample.txt")

if __name__ == "__main__":
    coloredlogs.install(level = "INFO")
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)
    #log.addHandler(utils.TQDMLoggingHandler())

    if torch.cuda.is_available() and args.cuda:
        device = torch.device("cuda") 
        logging.info("CUDA is being used")
    else:
        device = torch.device("cpu")
        logging.info("CPU is being used")

    logging.info(f"preparing dataset for {args.game}")

    # prepare data
    data_dir = args.data if args.data else "../data/smb/Processed/"

    if os.path.isdir(data_dir):
        logging.info(f"processing data from {data_dir}")

        mw = data.MW(data_dir)
        if args.game == "SMB1":
            mw.init_process_levels()

        levels = mw.read_level([i for i in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, i))])

        idx_to_col, col_to_idx = utils.tokenize(levels)
        logging.info(f"unique slices: {len(idx_to_col)} \t total slices: {len(np.concatenate(levels))}")

    else:
        raise Exception(f"directory {data_dir} does not exist!")

    # generate batches & datasets
    sequences, next_seq = data.gen_data(levels, args.step_size, args.seq_len)
    logging.info(f"sequences/batches: {len(sequences)}")
    logging.info(f"sequences.shape: {np.array(sequences).shape}")

    train_data, test_data = data.train_test_split(levels)
    train_data = data.batchify(train_data, 1)
    test_data = data.batchify(test_data, 1)

    # train!
    input_size = len(col_to_idx)
    try:
        logging.info(f"using {args.rnn_type}")
        train(train_data, test_data, args.rnn_type, args.nepochs, input_size, args.h, args.nl, args.w,
            args.lr, args.seq_len, args.bs, col_to_idx, idx_to_col, args.log_interval, args.game, device)
    except KeyboardInterrupt:
        print("EXITING EARLY!")
        exit()


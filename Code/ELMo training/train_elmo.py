
import argparse

import numpy as np

from bilm.training import train, load_options_latest_checkpoint, load_vocab
from bilm.data import BidirectionalLMDataset


def main(args):
    vocab = load_vocab(args.vocab_file, 50)

    batch_size = 1 # batch size for each GPU
    n_gpus = 3
    n_train_tokens = 4382

    options = {
        'bidirectional': True,

        "lstm": {"use_skip_connections": True, "projection_dim": 512, "cell_clip": 3, "proj_clip": 3, "dim": 1024,
              "n_layers": 2},
     "char_cnn": {"activation": "relu", "filters": [[1, 32], [2, 32], [3, 64], [4, 128], [5, 256], [6, 512], [7, 1024]],
                  "n_highway": 1, "embedding": {"dim": 512}, "n_characters": 261, "max_characters_per_token": 50},

        'dropout': 0.1,



        'all_clip_norm_val': 10.0,

        'n_epochs': 10,
        'n_train_tokens': n_train_tokens,
        'batch_size': batch_size,
        'n_tokens_vocab': vocab.size,
        'unroll_steps': 20,
        'n_negative_samples_batch': 8192,
    }


    prefix = args.train_prefix
    data = BidirectionalLMDataset(prefix, vocab, test=False,
                                      shuffle_on_load=True)

    tf_save_dir = args.save_dir
    tf_log_dir = args.save_dir
    train(options, data, n_gpus, tf_save_dir, tf_log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--vocab_file', help='Vocabulary file')
    parser.add_argument('--train_prefix', help='Prefix for train files')

    args = parser.parse_args()
    main(args)


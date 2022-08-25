# '''
# ELMo usage example to write biLM embeddings for an entire dataset to
# a file.
# '''
#
import os
import numpy as np
import h5py
from bilm import dump_bilm_embeddings

# Our small dataset.


# Create the dataset file.
dataset_file = 'tests/fixtures/train/cache/sentences_new.txt'

# Location of pretrained LM.  Here we use the test fixtures.
datadir = os.path.join('tests', 'fixtures', 'train','cache')
print(datadir)
vocab_file = os.path.join(datadir, 'vocab_hash.txt')
options_file = os.path.join(datadir, 'options.json')
weight_file = os.path.join(datadir, 'weights.hdf5')

# Dump the embeddings to a file. Run this once for your dataset.
embedding_file = 'tests/fixtures/train/cache/elmo_embeddings.hdf5'
dump_bilm_embeddings(
    vocab_file, dataset_file, options_file, weight_file, embedding_file
)

# Load the embeddings from the file -- here the 2nd sentence.
sentence_embeddings=[]
with h5py.File(embedding_file, 'r') as fin:
    print(fin.keys())
    second_sentence_embeddings = fin['1366'][...]
    second_sentence_embeddings[0]=second_sentence_embeddings[0]+second_sentence_embeddings[1]+second_sentence_embeddings[2]
    sentence_embeddings.append(second_sentence_embeddings[0]+second_sentence_embeddings[1]+second_sentence_embeddings[2])
    print(np.array(sentence_embeddings).shape)



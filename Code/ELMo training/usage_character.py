import tensorflow as tf
import os
from bilm import Batcher, BidirectionalLanguageModel, weight_layers

datadir = os.path.join('tests', 'fixtures', 'model')
vocab_file = os.path.join(datadir, 'vocab.txt')
options_file = os.path.join(datadir, 'options.json')
weight_file = os.path.join(datadir, 'lm_weights.hdf5')

batcher = Batcher(vocab_file, 50)

context_character_ids = tf.placeholder('int32', shape=(None, None, 50))

bilm = BidirectionalLanguageModel(options_file, weight_file)

context_embeddings_op = bilm(context_character_ids)

elmo_context_input = weight_layers('input', context_embeddings_op, l2_coef=0.0)

elmo_context_output = weight_layers(
    'output', context_embeddings_op, l2_coef=0.0
)

#raw_context=[]
raw_context = [
    'Pretrained biLMs compute representations useful for NLP tasks .',
    'They give state of the art performance for many tasks .'
]
tokenized_context = [sentence.split() for sentence in raw_context]
tokenized_question = [
    ['What', 'are', 'biLMs', 'useful', 'for', '?'],
]

# all_sentences='sentences.txt'
# with open(all_sentences, 'r') as lines:
#     for line in lines:
#         raw_context.append(line)

tokenized_context = [sentence.split() for sentence in raw_context]

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    context_ids = batcher.batch_sentences(tokenized_context)
    question_ids = batcher.batch_sentences(tokenized_question)
    
    elmo_context_input_, elmo_question_input_ = sess.run(
        [elmo_context_input['weighted_op'], elmo_question_input['weighted_op']],
        feed_dict={context_character_ids: context_ids,
                   question_character_ids: question_ids}
    )


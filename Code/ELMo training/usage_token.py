from sklearn.decomposition import PCA
from matplotlib import pyplot
import tensorflow as tf
import os
from bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers, \
    dump_token_embeddings

raw_context = [
    'Pretrained biLMs compute representations useful for NLP tasks .',
    'They give state of the art performance for many tasks .'
]

# raw_context=[]
# all_sentences='sentences.txt'
# with open(all_sentences, 'r') as lines:
#     for line in lines:
#         raw_context.append(line)

tokenized_context = [sentence.split() for sentence in raw_context]

tokenized_question = [
    ['What', 'are', 'biLMs', 'useful', 'for', '?'],
]
words=[]
word_list=tokenized_question
for sentence in word_list:
    for word in sentence:
        words.append(word)

all_tokens = set(['<S>', '</S>'])

for context_sentence in tokenized_context:
    for token in context_sentence:
        all_tokens.add(token)
vocab_file = 'vocab.txt'
with open(vocab_file, 'w') as fout:
    fout.write('\n'.join(all_tokens))

    
datadir = os.path.join('tests', 'token')
options_file = os.path.join(datadir, 'options.json')
weight_file = os.path.join(datadir, 'weights.hdf5')


token_embedding_file = 'elmo_token_embeddings.hdf5'
print("token_embedding_file",token_embedding_file)
dump_token_embeddings(
    vocab_file, options_file, weight_file, token_embedding_file
)
tf.reset_default_graph()

batcher = TokenBatcher(vocab_file)

context_token_ids = tf.placeholder('int32', shape=(None, None))

bilm = BidirectionalLanguageModel(
    options_file,
    weight_file,
    use_character_inputs=False,
    embedding_weight_file=token_embedding_file
)

context_embeddings_op = bilm(context_token_ids)

elmo_context_input = weight_layers('input', context_embeddings_op, l2_coef=0.0)


elmo_context_output = weight_layers(
    'output', context_embeddings_op, l2_coef=0.0
)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    context_ids = batcher.batch_sentences(tokenized_context)

    elmo_context_input_ = sess.run(
        [elmo_context_input['weighted_op']],
        feed_dict={context_token_ids: context_ids
                   }
    )

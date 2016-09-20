from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import urllib
import urllib2
import tarfile
import os
import random
import string
import cPickle
from Helper import Helper

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# a parameter, hard coded, to represent the
# size of the vocab. This can be changed if necessary...
vocabulary_size = 50000


def word2vec():
    # hyperparameters - TODO - place into FLAGS (tensorflow website has examples)
    batch_size = 128 # how many target/context words to get in each batch
    embedding_size = 128  # Dimension of the embedding vector.
    skip_window = 1  # How many words to consider left and right - context size
    num_skips = 2  # How many times to reuse an input to generate a label
    # TAKEN FROM TF WEBSITE EXAMPLE:
    # We pick a random validation set to sample nearest neighbors. here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent. 
    valid_size = 16  # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.array(random.sample(range(valid_window), valid_size))
    num_sampled = 64# Number of negative examples to sample.

    num_steps = 50001  # steps to run for
    steps_per_checkpoint = 50 # save the params every 50 steps.

    # prep work
    basedir = os.getcwd()

    #pull the data and get it into a usable format.
    get_imdb_data(basedir)


    data, count, dictionary, reverse_dictionary = build_dataset(basedir)
    # save the dictionary to file - very important for Data Processor
    Helper.store_stuff(dictionary, "dictionary.pickle", reverse_dictionary, "reverse_dictionary.pickle")
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10])
    batch_tester(data, reverse_dictionary)
    print('three index', dictionary['three'])

    ckpt_path = os.path.join(basedir, 'checkpoints')
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    ckpt_embed = os.path.join(ckpt_path, "embeddings")
    if not os.path.exists(ckpt_embed):
        os.makedirs(ckpt_embed)
    # actual computation
    # TODO refactor this! 
    # could follow guidelines here https://danijar.com/structuring-your-tensorflow-models/
    graph = tf.Graph()
    with graph.as_default():
        # variable to track progress
        global_step = tf.Variable(0, trainable=False)

        # Input data.
        train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
        with tf.device('/cpu:0'):
            # Variables.
            embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), 
                                    name = "embeddings")
            nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

            # Model.
            # Look up embeddings for inputs.
            # note that the embeddings are Variable params that will
            # be optimised!
            embed = tf.nn.embedding_lookup(embeddings, train_dataset)
        # Compute the nce loss, using a sample of the negative labels each time.
        # tried using sampled_softmax_loss, but performance was worse, so decided
        # to use NCE loss instead. Might be worth some more testing, especially with
        # the hyperparameters (ie num_sampled), to see what gives the best performance.
        # tuning these params is a TODO.
        loss = tf.reduce_mean(
            tf.nn.nce_loss(nce_weights, nce_biases, embed,
                           train_labels, num_sampled, vocabulary_size))

        # PART BELOW LIFTED FROM TF EXAMPLES
        # Optimizer.
        # Note: The optimizer will optimize the nce weights AND the embeddings.
        # This is because the embeddings are defined as a variable quantity and the
        # optimizer's `minimize` method will by default modify all variable quantities 
        # that contribute to the tensor it is passed.
        # See docs on `tf.train.Optimizer.minimize()` for more details.
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss, global_step=global_step)

        # Compute the similarity between minibatch examples and all embeddings.
        # We use the cosine distance:
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(
            normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

    # This helps us terminate early if training started before.
    started_before = False
    with tf.Session(graph=graph) as session:
        # want to save the overall state and the embeddings for later.
        # I think we can do this in one, but I haven't had time to test this yet.
        # TODO make this a bit more efficient, avoid having to save stuff twice.
        # NOTE - this part is very closely coupled with the lstm.py script, as it
        # reads the embeddings from the location specified here. Might be worth
        # relaxing this dependency and passing the save location as a variable param.
        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        saver = tf.train.Saver(tf.all_variables())
        saver_embed = tf.train.Saver({'embeddings': embeddings})
        if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from {0}".format(ckpt.model_checkpoint_path))
            saver.restore(session, ckpt.model_checkpoint_path)
            print("done")
            started_before = True
        else:
            print("Creating model with fresh parameters.")
            tf.initialize_all_variables().run()
            print('Initialized')
        average_loss = 0
        for step in range(num_steps):
            batch_data, batch_labels = generate_batch(data, batch_size, num_skips, skip_window)
            feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
            _, l = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += l

            if step >= 10000 and (average_loss / 2000) < 5 and started_before:
                print('early finish as probably loaded from earlier')
                break

            if step % steps_per_checkpoint == 0:
                # save stuff
                checkpoint_path = os.path.join(ckpt_path, "model_ckpt")
                embed_path = os.path.join(ckpt_embed,"embeddings_ckpt")
                saver.save(session, checkpoint_path, global_step=global_step)
                saver_embed.save(session, embed_path)
            if step % 2000 == 0:
                if step > 0:
                    average_loss = average_loss / 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step %d: %f' % (step, average_loss))
                average_loss = 0

                # note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 10000 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log = '%s %s,' % (log, close_word)
                    print(log)

        final_embeddings = normalized_embeddings.eval()
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        plot_only = 500
        low_dim_embs = tsne.fit_transform(final_embeddings[1:plot_only+1, :])
        labels = [reverse_dictionary[i] for i in range(plot_only)]
        plot_with_labels(low_dim_embs, labels)

        # from here, need to take the embedding parameters and then pass them to the next stage of the
        # system - the sentiment analyser
        # ie , save final_embeddings. This has been done in the actual operation.


# get the data!
# if not downloaded already, download the tarfile and extract it.
# Then, lump ALL the data from the reviews together into 1 big file,
# called alldata.txt. This makes the model's operation much easier.
def get_imdb_data(basedir):
    url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    imdb_path = os.path.join(basedir, 'imdb_data')
    docname = 'imdb_data.tar'
    if not os.path.exists(imdb_path):
        os.makedirs(imdb_path)
        testfile = urllib.URLopener()
        try:
            testfile.retrieve(url, docname)
            tar = tarfile.open(docname, "r")
            tar.extractall(imdb_path)
            tar.close()
        except (urllib2.URLError, tarfile.ReadError):
            exit()
    make_one_file(imdb_path, basedir)



# Take all the files associated with the training set,
# both +ve and -ve reviews. for each of these files,
# open it and read the contents into another file, all_data.txt
# Skip the unsupervised learning examples for now - could
# include them at a later data
def make_one_file(imdb_path, basedir):
    training_set = os.path.join(imdb_path, 'aclImdb/train')
    target = os.path.join(basedir, 'all_data.txt')
    if os.path.exists(target):
        return
    li = os.listdir(training_set)
    print(li)
    with open(os.path.join(target), "w") as fo:
        for item in li:
            item = os.path.join(training_set, item)
            print(item)
            if os.path.isdir(item) and item.find("unsup")==-1: # hack to avoid unsupervised folder
                print('found directory', item)
                reviews = os.listdir(item)
                for review in reviews:
                    review = os.path.join(item, review)
                    with open(review, 'r') as fi:
                        fo.write(fi.read())
                        fo.write('\n')


# reads the words from all_data.txt into a buffer.
# takes this data, removes unwanted tags
# also (for now) removes ALL punctuation
# returns data as a list of words.
# TODO assess whether removal of ALL punctuation is good
# - might not matter as long as we're consistent throughout the system.
def get_words(basedir):
    table = string.maketrans("", "")
    with open(os.path.join(basedir, 'all_data.txt'), "r") as fi:
        # test this?
        data = tf.compat.as_str(fi.read())
    li = data.split('<br /><br />')
    tmp = " ".join(li)
    tmp = tmp.translate(table, string.punctuation).lower()
    li = tmp.split()
    return li


# get all the words as a list.
# then, take the most common words, and add them to a new list of tuples
# first is the word as a string, second is the number of times it occurs
# in the list of all words.
# This then builds a dictionary containing all the words seen as keys, 
# and values as their occurrence rank  in the entire body of text - 
# see below.
# finally, it tokenizes the words in the big file, so we have an integer lookup for each.
# Key point to note is that that the first entry in the dictionary is the placeholder
# "<PAD>", which is necessary for the passing the sequence length into the 
# LSTM for sentiment analysis.
def build_dataset(basedir):
    words = get_words(basedir)
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 2))
    dictionary = dict()
    dictionary["<PAD>"] = len(dictionary) # store this first - helps for zero padding later
    for word, _ in count:
        # dictionary contains words as keys, values are the occurrence rank
        # ie most common word has value 1
        # second value two
        # etc
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count = unk_count + 1
        # the index of the word (unique token) is equal to its occurrence rank
        # data contains all the words' unique tokens in order, as they appear
        # in the text file with all the reviews
        data.append(index)
    count[0][1] = unk_count
    # reverse dict enables us to go from token for word (occurrence rank)
    # to the actual word
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    del words
    return data, count, dictionary, reverse_dictionary


# This is a global index into the dataset.
data_index = 0

# This was lifted straight from the example - operation
# is a little mysterious, but it effectively just generates
# a series of context and target words for the model to train on.
def generate_batch(data, batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=batch_size, dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


# just a test to check that things are working as they should.
# mainly for debugging purposes - can be removed at a later data if need be.
def batch_tester(data, reverse_dictionary):
    print('data:', [reverse_dictionary[di] for di in data[:8]])
    for num_skips, skip_window in [(2, 1), (4, 2)]:
        batch, labels = generate_batch(data, batch_size=8, num_skips=num_skips, skip_window=skip_window)
        print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
        print('    batch:', [reverse_dictionary[bi] for bi in batch])
        print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        x = round(x,4)
        y = round(y,4)
        plt.scatter(x, y)
        plt.annotate(unicode(label,"utf-8"),
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)


if __name__ == '__main__':
    word2vec()

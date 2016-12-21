import string
import os
import numpy as np
import cPickle
from threading import Thread, Lock

from Helper import Helper

# the directories where all the imdb data is located. This is hardcoded in for now,
# but it relies on what's been set up by word2vec.py script. Could relax this dependency later.
dirs = ["imdb_data/aclImdb/test/pos", "imdb_data/aclImdb/test/neg", "imdb_data/aclImdb/train/pos",
        "imdb_data/aclImdb/train/neg"]

# the max allowable  length for a review.
# chosen by using mean + 2*std_dev, from
# https://cs224d.stanford.edu/reports/HongJames.pdf
max_seq_length = 650

li_train = []
li_test = []

# main entry point for the script.
# take all the reviews - convert them into numpy arrays.
# also take the LSTM sources and convert the parts
# mentioning a keyword into np arrays 
def build_data():
    if not os.path.exists("imdb_data/processed"):
        os.makedirs("imdb_data/processed/")
        print("No processed data found - generating")
    else:
        print("Some data may exist!")

    # hardcoded, based on word2vec. TODO Could use some freeing up!
    word_ids, rev_word_ids = Helper.get_stuff("dictionary.pickle", "reverse_dictionary.pickle")

    dir_count = 0
    processes = []
    # li_train and li_test are shared memory locations
    # that contain the filepaths of the processed articles.
    # this is why access is controlled by a lock.
    li_test_lock = Lock()
    li_train_lock = Lock()
    stdout_lock = Lock()
    for d in dirs:
        print("Procesing data with process: " + str(dir_count))
        if d.find("test") != -1:
            p = Thread(target=create_processed_files,
                        args=(dir_count, word_ids, d, max_seq_length, "test", li_test_lock, stdout_lock))
        else:
            p = Thread(target=create_processed_files,
                        args=(dir_count, word_ids, d, max_seq_length, "train", li_train_lock, stdout_lock))
        p.start()
        processes.append(p)
        dir_count += 1
    for p in processes:
        if p.is_alive():
            p.join()

    # very important to store these so that LSTM can use them.
    Helper.store_stuff(li_train, "li_train.pickle", li_test, "li_test.pickle")


def create_processed_files(pid, word_ids, direc, max_seq_length, datatype, li_lock, stdout_lock):
    # first get the label for the data - is it a +ve/-ve example?
    if direc.find("pos") != -1:
        label = 1
    else:
        label = 0
    global li_train
    global li_test
    count = 0
    internal_list = []
    # objective - convert a series of words in each text file into a numpy
    # integer array, corresponding to word tokens from the word_ids dictionary
    # ie - tokenise each word in the review. the dictionary word_ids is very helpful for this.
    for review in os.listdir(direc):
        count += 1
        if count % 100 == 0:
            stdout_lock.acquire()
            print ("Processing: " + review + " the " + str(count) + "th file... in process number" + str(pid))
            stdout_lock.release()
        # for a given review, output file is a string with the filepath of that tokenised review
        output_file = process_review(os.path.join(direc, review), word_ids, max_seq_length, datatype)
        if output_file is not None:
            internal_list.append((output_file, label))

    # at the end, take the internal list and add it to the global list
    # then clear the internal list
    # TODO assess the efficiency of this,no idea whether it's good or not
    # for example, should we instead be doing this after N steps???
    li_lock.acquire()
    stdout_lock.acquire()

    print (internal_list[:10])
    if datatype == "test":
        li_test.extend(internal_list)
    else:
        li_train.extend(internal_list)
    stdout_lock.release()
    li_lock.release()

def process_review(review_filepath, word_ids, max_seq_length, data_type):
    # take a review at review_filepath, convert it into a numpy array of scalars
    # pad the end with zeros if necessary
    # the padding step is very important - it allows us to deduce the length of the
    # review, and pass this as a seq_length argument to the LSTM when it does
    # sentiment analysis.
    # store it in a new file- return the filepath to calling function
    data = np.array([i for i in range(max_seq_length)])
    word_indices = []

    # handle case where file already exists - don't process again!
    processed_dir = os.path.join("imdb_data/processed",data_type)
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    new_name = Helper.remove_file_extension(process(review_filepath)) + ".npy"
    processed_path = os.path.join(processed_dir, new_name)
    if os.path.exists(processed_path):
        return processed_path

    with open(review_filepath, 'r') as review:
        # first read the file contents into a buffer
        content = review.read().lower()
    # strip the punctuation and unnecessary bits - TODO put this into Helper class
    # get back a list of words
    word_list = cleanup(content)
    for word in word_list:
        if word in word_ids:
            token = word_ids[word]
        else:
            token = word_ids["UNK"]
        word_indices.append(token)
    # post processing
    # pad sequence to max length
    if len(word_indices) < max_seq_length:
        padding = word_ids["<PAD>"]
        assert(padding == 0) # useful sanity check - important for later.
        # could use a negative number if need be, but it's not that important
        word_indices = word_indices + [padding for i in range(max_seq_length - len(word_indices))]
    else:
        word_indices = word_indices[0:max_seq_length]
    data = np.vstack((data, word_indices))
    # data is set up to have the right size, but the first row is just full of dummy data.
    # this slicing procedure extracts JUST the word indices.
    data = data[1::]
    save_data(data, processed_path)
    return processed_path

def cleanup(text):
    table = string.maketrans("", "")
    # reads the words from all_data.txt into a buffer
    li = text.split('<br /><br />')
    tmp = " ".join(li)
    tmp = tmp.translate(table, string.punctuation).lower()
    li = tmp.split()
    return li

def save_data(npArray, path):
    start = os.getcwd()
    path = os.path.join(start, path)
    np.save(path, npArray)


# TODO move this into helper class...
def process(name):
    indices = [i for i, ltr in enumerate(name) if ltr == '/']
    return name[indices[-1] + 1:]


if __name__ == '__main__':
    build_data()

# Inspired from https://github.com/deborausujono/word2vecpy/blob/master/word2vec.py
# Mofified for INF8225 final project purpuses: Ahmed Hammami
#                                              Andrei Catana
#                                              Fabrice Simo Defo
#                                              Fatou S.Mouenzeo

import argparse
import math
import struct
import sys
import time
import warnings
import copy
import os
import numpy as np
import urllib.request
import zipfile
import pickle
import matplotlib.pyplot as plt
import re
from nltk.tokenize import RegexpTokenizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors 
from multiprocessing import Pool, Value, Array

class Utils:
    def __init(self):
        pass

    def download_data(self, url="http://mattmahoney.net/dc/enwik8.zip", expected_bytes=36445475, local_data_dir='data/'):
        """
        :param url: data download address
        :param expected_bytes: expected size of data
        :param local_data_dir: local dir to save data
        :return: path of downloaded file
        """
        if not os.path.exists(local_data_dir):
            os.makedirs(local_data_dir)
        local_data_path = local_data_dir + os.path.split(url)[1]

        if not os.path.exists(local_data_path):
            print("Downloading training set {}\n".format(url))
            urllib.request.urlretrieve(url, local_data_path)
        if os.stat(local_data_path).st_size == expected_bytes:
            print('Found and verified', local_data_path)
        else:
            raise Exception(local_data_path, ' is found, but the size is not right. ',
                            'found_bytes', os.stat(local_data_path).st_size, ', expected_bytes', expected_bytes,
                            'Can you get to it with a browser?')

        with zipfile.ZipFile(local_data_path, 'r') as zip_ref:
            zip_ref.extractall(local_data_dir)

        return local_data_path[:-4]

    def preprocess_question_words(self, vocab_list):
        """
        :param vocab_list: word list
        :return: Generates files for semantic and syntactic model test
        """
        file_name = 'data/' + 'questions-words.txt'
        with open(file_name) as f:
            fw1 = open("data/" + 'semantic.txt', 'w', encoding='utf-8')
            fw2 = open("data/" + 'syntactic.txt', 'w', encoding='utf-8')
            fw1.write(": semantic annotations\n")
            fw2.write(": syntactic annotations\n")

            fw = fw1
            for i, line in enumerate(f.readlines()):
                if line[0] == ':':
                    fw = fw2 if line[2:6] == 'gram' else fw1
                else:
                    words = line.strip().split()
                    need_write = True
                    for word in words:
                        if word not in vocab_list:
                            need_write = False
                            break

                    if need_write == True:
                        fw.write(line)
        print('preprocess over')

    def load_model(self, path_to_model):
        """
        :param path_to_model: the path for the trained model
        :return: loaded model from the file as a dict
        """
        print("Loading model from file {}".format(path_to_model))
        words = dict()
        with open(path_to_model, "r") as file:
            file.readline()
            for l in file:
                line = l.split(" ")
                word = line.pop(0)
                line = [float(ll) for ll in line]
                words[word] = np.array(line)
        
        return words

    def process_pca(self, words_to_plot, words, save_path):
        """
        :param words_to_plot: the only words showing in the pca final plot
        :param words: dict of words and their embeddings
        :param save_path: path to save the figure
        Plotting the pca for specific words
        """
        vectors_index_to_plot = []
        for item in words_to_plot:
            vectors_index_to_plot.append(list(words.keys()).index(item))

        M = np.array(list(words.values()))
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(M)
        t = reduced.transpose()
        
        t_0 = []
        t_1 = []
        for i in range(0, len(vectors_index_to_plot)):
            plt.annotate(words_to_plot[i], (t[0][vectors_index_to_plot[i]], t[1][vectors_index_to_plot[i]]))
            t_0.append(t[0][vectors_index_to_plot[i]])
            t_1.append(t[1][vectors_index_to_plot[i]])
        plt.scatter(t_0, t_1)
        plt.savefig(save_path)
        # plt.show()
        plt.close()
        print("PCA saved to {}".format(save_path))

    def process_words_associations(self, words, save_path, plot_only=100):
        """
        :param words: dict of words and their embeddings
        :param save_path: path to save the figure
        :param plot_only: number of words to present in the final graph
        Plotting the pca for specific words
        """
        M = np.array(list(words.values()))
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)

        two_d_embeddings = tsne.fit_transform(M[:plot_only, :])
        num_points = len(two_d_embeddings)

        words_labels = [list(words.keys())[i] for i in range(num_points)]

        plt.figure(figsize=(16, 9))
        for i, label in enumerate(words_labels):
            x, y = two_d_embeddings[i,:]
            plt.scatter(x, y)
            plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
        plt.savefig(save_path)
        # plt.show()
        plt.close()
        print("words associations saved to {}".format(save_path))

    def test_model(self, path_to_model):
        """
        :param path_to_model: the path for the trained model
        :return: Generate PCA, words vector in space and accuracies tests for the model in path
        """
        
        words = self.load_model(path_to_model)
        model = KeyedVectors.load_word2vec_format(path_to_model)

        print("\n**********************\n10 most probable words for the vector woman - Queen + man\n")
        result = model.most_similar(positive=['woman', 'Queen'], negative=['man'])
        for i,x in enumerate(result):
            print("{}: {:.4f}".format(*result[i]))

        print("\n**********************\n10 most probable words for the vector France - Italy + Paris\n")
        result = model.most_similar(positive=['France', 'Paris'], negative=['Italy'])
        for i,x in enumerate(result):
            print("{}: {:.4f}".format(*result[i]))

        semantic_accuracy = model.evaluate_word_analogies(analogies='data/semantic.txt', case_insensitive=False)[0]
        print("semantic accuracy: {}".format(semantic_accuracy))
        syntactic_accuracy = model.evaluate_word_analogies(analogies='data/syntactic.txt', case_insensitive=False)[0]
        print("syntactic accuracy: {}".format(syntactic_accuracy))

        words_to_plot = [
            'China', 'Beijing',
            'Russia', 'Moscow',
            'Japan', 'Tokyo',
            'Turkey', 'Ankara',
            'Poland', 'Warsaw',
            'Germany', 'Berlin',
            'France', 'Paris',
            'Italy', 'Rome',
            'Greece', 'Athens',
            'Spain', 'Madrid',
            'Portugal', 'Lisbon'
        ]
        self.process_pca(words_to_plot, words, "pca.png")

        self.process_words_associations(words, "words_associations.png", plot_only=50)


class VocabItem:
    def __init__(self, word):
        self.word = word
        self.count = 0
        self.path = None # Path (list of indices) from the root to the word (leaf)
        self.code = None # Huffman encoding

class Vocab:
    def __init__(self, fi, min_count):
        vocab_items = []
        vocab_hash = {}
        word_count = 0
        fi = open(fi, 'r')
        tokenizer = RegexpTokenizer(r'\w+')
        # Add special tokens <bol> (beginning of line) and <eol> (end of line)
        for token in ['<bol>', '<eol>']:
            vocab_hash[token] = len(vocab_items)
            vocab_items.append(VocabItem(token))

        for line in fi:
            tokens = tokenizer.tokenize(line)
            for token in tokens:
                pattern = re.compile("[a-zA-Z]")
                if pattern.match(token):
                    if token not in vocab_hash:
                        vocab_hash[token] = len(vocab_items)
                        vocab_items.append(VocabItem(token))
                        
                    #assert vocab_items[vocab_hash[token]].word == token, 'Wrong vocab_hash index'
                    vocab_items[vocab_hash[token]].count += 1
                    word_count += 1
                
                    if word_count % 10000 == 0:
                        sys.stdout.write("\rReading word %d" % word_count)
                        sys.stdout.flush()

            # Add special tokens <bol> (beginning of line) and <eol> (end of line)
            vocab_items[vocab_hash['<bol>']].count += 1
            vocab_items[vocab_hash['<eol>']].count += 1
            word_count += 2

        self.bytes = fi.tell()
        self.vocab_items = vocab_items         # List of VocabItem objects
        self.vocab_hash = vocab_hash           # Mapping from each token to its index in vocab
        self.word_count = word_count           # Total number of words in train file

        # Filling semantic and syntaxinc files for accuracy
        print("\nfilling syntactic and semantic test cases for the training set loaded from {}".format(fi))
        Utils().preprocess_question_words(list(self.vocab_hash.keys()))

        # Add special token <unk> (unknown),
        # merge words occurring less than min_count into <unk>, and
        # sort vocab in descending order by frequency in train file
        self.__sort(min_count)

        #assert self.word_count == sum([t.count for t in self.vocab_items]), 'word_count and sum of t.count do not agree'
        print('Total words in training file: %d' % self.word_count)
        print('Total bytes in training file: %d' % self.bytes)
        print('Vocab size: %d' % len(self))

    def __getitem__(self, i):
        return self.vocab_items[i]

    def __len__(self):
        return len(self.vocab_items)

    def __iter__(self):
        return iter(self.vocab_items)

    def __contains__(self, key):
        return key in self.vocab_hash

    def __sort(self, min_count):
        tmp = []
        tmp.append(VocabItem('<unk>'))
        unk_hash = 0
        
        count_unk = 0
        for token in self.vocab_items:
            if token.count < min_count:
                count_unk += 1
                tmp[unk_hash].count += token.count
            else:
                tmp.append(token)

        tmp.sort(key=lambda token : token.count, reverse=True)

        # Update vocab_hash
        vocab_hash = {}
        for i, token in enumerate(tmp):
            vocab_hash[token.word] = i

        self.vocab_items = tmp
        self.vocab_hash = vocab_hash

        print()
        print('Unknown vocab size:', count_unk)

    def indices(self, tokens):
        return [self.vocab_hash[token] if token in self else self.vocab_hash['<unk>'] for token in tokens]

    def encode_huffman(self):
        # Build a Huffman tree
        vocab_size = len(self)
        count = [t.count for t in self] + [1e15] * (vocab_size - 1)
        parent = [0] * (2 * vocab_size - 2)
        binary = [0] * (2 * vocab_size - 2)
        
        pos1 = vocab_size - 1
        pos2 = vocab_size

        for i in range(vocab_size - 1):
            # Find min1
            if pos1 >= 0:
                if count[pos1] < count[pos2]:
                    min1 = pos1
                    pos1 -= 1
                else:
                    min1 = pos2
                    pos2 += 1
            else:
                min1 = pos2
                pos2 += 1

            # Find min2
            if pos1 >= 0:
                if count[pos1] < count[pos2]:
                    min2 = pos1
                    pos1 -= 1
                else:
                    min2 = pos2
                    pos2 += 1
            else:
                min2 = pos2
                pos2 += 1

            count[vocab_size + i] = count[min1] + count[min2]
            parent[min1] = vocab_size + i
            parent[min2] = vocab_size + i
            binary[min2] = 1

        # Assign binary code and path pointers to each vocab word
        root_idx = 2 * vocab_size - 2
        for i, token in enumerate(self):
            path = [] # List of indices from the leaf to the root
            code = [] # Binary Huffman encoding from the leaf to the root

            node_idx = i
            while node_idx < root_idx:
                if node_idx >= vocab_size: path.append(node_idx)
                code.append(binary[node_idx])
                node_idx = parent[node_idx]
            path.append(root_idx)

            # These are path and code from the root to the leaf
            token.path = [j - vocab_size for j in path[::-1]]
            token.code = code[::-1]

class UnigramTable:
    """
    A list of indices of tokens in the vocab following a power law distribution,
    used to draw negative samples.
    """
    def __init__(self, vocab):
        vocab_size = len(vocab)
        power = 0.75
        norm = sum([math.pow(t.count, power) for t in vocab]) # Normalizing constant

        table_size = int(1e8) # Length of the unigram table
        table = np.zeros(table_size, dtype=np.uint32)

        print('Filling unigram table')
        p = 0 # Cumulative probability
        i = 0
        for j, unigram in enumerate(vocab):
            p += float(math.pow(unigram.count, power))/norm
            while i < table_size and float(i) / table_size < p:
                table[i] = j
                i += 1
        self.table = table

    def sample(self, count):
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i] for i in indices]

def sigmoid(z):
    if z > 6:
        return 1.0
    elif z < -6:
        return 0.0
    else:
        return 1 / (1 + math.exp(-z))

def init_net(dim, vocab_size):
    # Init syn0 with random numbers from a uniform distribution on the interval [-0.5, 0.5]/dim
    tmp = np.random.uniform(low=-0.5/dim, high=0.5/dim, size=(vocab_size, dim))
    syn0 = np.ctypeslib.as_ctypes(tmp)
    syn0 = Array(syn0._type_, syn0, lock=False)

    # Init syn1 with zeros
    tmp = np.zeros(shape=(vocab_size, dim))
    syn1 = np.ctypeslib.as_ctypes(tmp)
    syn1 = Array(syn1._type_, syn1, lock=False)

    return (syn0, syn1)

def train_process(pid):
    # Set fi to point to the right chunk of training file
    start = vocab.bytes / num_processes * pid
    end = vocab.bytes if pid == num_processes - 1 else vocab.bytes / num_processes * (pid + 1)
    global_word_count_copy = global_word_count.value

    for e in range(epochs):
        fi.seek(start)
        #print 'Worker %d beginning training at %d, ending at %d' % (pid, start, end)

        alpha = starting_alpha
        global_word_count.value = global_word_count_copy
        word_count = 0
        last_word_count = 0
        while fi.tell() < end:
            line = fi.readline().strip()
            # Skip blank lines
            if not line:
                continue

            # Init sent, a list of indices of words in line
            sent = vocab.indices(['<bol>'] + line.split() + ['<eol>'])

            for sent_pos, token in enumerate(sent):
                if word_count % 10000 == 0:
                    global_word_count.value += (word_count - last_word_count)
                    last_word_count = word_count

                    # Recalculate alpha
                    alpha = starting_alpha * (1 - float(global_word_count.value) / vocab.word_count)
                    if alpha < starting_alpha * 0.0001: alpha = starting_alpha * 0.0001

                    # Print progress info
                    sys.stdout.write("\rAlpha: %f Progress: %d of %d (%.2f%%)" %
                                    (alpha, global_word_count.value, vocab.word_count,
                                    float(global_word_count.value) / vocab.word_count * 100))
                    sys.stdout.flush()

                # Randomize window size, where win is the max window size
                current_win = np.random.randint(low=1, high=win+1)
                context_start = max(sent_pos - current_win, 0)
                context_end = min(sent_pos + current_win + 1, len(sent))
                context = sent[context_start:sent_pos] + sent[sent_pos+1:context_end] # Turn into an iterator?

                # CBOW
                if cbow:
                    # Compute neu1
                    neu1 = np.mean(np.array([syn0[c] for c in context]), axis=0)
                    assert len(neu1) == dim, 'neu1 and dim do not agree'

                    # Init neu1e with zeros
                    neu1e = np.zeros(dim)

                    # Compute neu1e and update syn1
                    if neg > 0:
                        classifiers = [(token, 1)] + [(target, 0) for target in table.sample(neg)]
                    else:
                        classifiers = zip(vocab[token].path, vocab[token].code)
                    for target, label in classifiers:
                        z = np.dot(neu1, syn1[target])
                        p = sigmoid(z)
                        g = alpha * (label - p)
                        neu1e += g * syn1[target] # Error to backpropagate to syn0
                        syn1[target] += g * neu1  # Update syn1

                    # Update syn0
                    for context_word in context:
                        syn0[context_word] += neu1e

                # Skip-gram
                else:
                    for context_word in context:
                        # Init neu1e with zeros
                        neu1e = np.zeros(dim)

                        # Compute neu1e and update syn1
                        if neg > 0:
                            classifiers = [(token, 1)] + [(target, 0) for target in table.sample(neg)]
                        else:
                            classifiers = zip(vocab[token].path, vocab[token].code)
                        for target, label in classifiers:
                            z = np.dot(syn0[context_word], syn1[target])
                            p = sigmoid(z)
                            g = alpha * (label - p)
                            neu1e += g * syn1[target]              # Error to backpropagate to syn0
                            syn1[target] += g * syn0[context_word] # Update syn1

                        # Update syn0
                        syn0[context_word] += neu1e

                word_count += 1

        print("epoch {} from worker {}\n".format(e, pid))

        # Print progress info
        global_word_count.value += (word_count - last_word_count)
        sys.stdout.write("\rAlpha: %f Progress: %d of %d (%.2f%%)" %
                        (alpha, global_word_count.value, vocab.word_count,
                        float(global_word_count.value)/vocab.word_count * 100))
        sys.stdout.flush()
    fi.close()

def save(vocab, syn0, fo):
    print('Saving model to', fo)
    dim = len(syn0[0])

    fo = open(fo, 'w')
    fo.write('%d %d\n' % (len(syn0), dim))
    for token, vector in zip(vocab, syn0):
        word = token.word
        vector_str = ' '.join([str(s) for s in vector])
        fo.write('%s %s\n' % (word, vector_str))

    fo.close()

def __init_process(*args):
    global vocab, syn0, syn1, table, cbow, neg, dim, starting_alpha
    global win, num_processes, global_word_count, fi, epochs
    
    vocab, syn0_tmp, syn1_tmp, table, cbow, neg, dim, starting_alpha, win, num_processes, global_word_count, epochs = args[:-1]
    fi = open(args[-1], 'r')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        syn0 = np.ctypeslib.as_array(syn0_tmp)
        syn1 = np.ctypeslib.as_array(syn1_tmp)

def train(fi, fo, epochs, cbow, neg, dim, alpha, win, min_count, num_processes):

    # Read train file to init vocab
    if fi == "data/enwik8":
        Utils().download_data()

    vocab = Vocab(fi, min_count)

    # Init net
    syn0, syn1 = init_net(dim, len(vocab))

    
    global_word_count = Value('i', 0)
    table = None
    if neg > 0:
        print('Initializing unigram table')
        table = UnigramTable(vocab)
    else:
        print('Initializing Huffman tree')
        vocab.encode_huffman()

    # Begin training using num_processes workers
    t0 = time.time()
    pool = Pool(processes=num_processes, initializer=__init_process,
                initargs=(vocab, syn0, syn1, table, cbow, neg, dim, alpha,
                          win, num_processes, global_word_count, epochs, fi))
    pool.map(train_process, range(num_processes))
    t1 = time.time()
    print()
    print('Completed training. Training took', (t1 - t0) / 60, 'minutes')

    # Save model to file
    save(vocab, syn0, fo)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-test', help='Load trained model and test it', dest='test', default=False, type=bool)
    parser.add_argument('-model', help='Output model file -> if test is enabled, then provide the path for the model to test', dest='fo', required=True)
    parser.add_argument('-train', help='Training file', dest='fi', default='data/enwik8')
    parser.add_argument('-cbow', help='1 for CBOW, 0 for skip-gram', dest='cbow', default=1, type=int)
    parser.add_argument('-negative', help='Number of negative examples (>0) for negative sampling, 0 for hierarchical softmax', dest='neg', default=5, type=int)
    parser.add_argument('-dim', help='Dimensionality of word embeddings', dest='dim', default=1000, type=int)
    parser.add_argument('-alpha', help='Starting alpha', dest='alpha', default=0.025, type=float)
    parser.add_argument('-window', help='Max window length', dest='win', default=5, type=int) 
    parser.add_argument('-min-count', help='Min count for words used to learn <unk>', dest='min_count', default=5, type=int)
    parser.add_argument('-processes', help='Number of processes', dest='num_processes', default=1, type=int)
    parser.add_argument('-epochs', help='Number of training epochs', dest='epochs', default=1, type=int)
    args = parser.parse_args()

    if args.test:
        Utils().test_model(args.fo)
    else:
        train(args.fi, args.fo, args.epochs, bool(args.cbow), args.neg, args.dim, args.alpha, args.win,
          args.min_count, args.num_processes)

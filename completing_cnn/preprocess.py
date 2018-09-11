# -*- encoding:utf-8 -*-
import numpy as np
import gensim
import codecs
import numpy.random as random
from data_helpers import clean_str

class Word2Vec():
    def __init__(self):
        self.model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin',
        #self.model = gensim.models.KeyedVectors.load_word2vec_format('./vectors.bin',
                                                                     binary=True)
        self.unknowns = np.random.uniform(-0.01, 0.01, 300).astype("float32")

    def get(self, word):
        if word not in self.model.vocab:
            return self.unknowns
        else:
            return self.model.word_vec(word)


class Data():
    def __init__(self, word2vec, max_len=39, shuffle=True):
        self.s, self.labels = [], []
        self.index, self.max_len, self.word2vec = 0, max_len, word2vec
        self.shuffle = shuffle
  
    def open_file(self):
        pass

    def is_available(self):
        if self.index < self.data_size:
            return True
        else:
            return False

    def reset_index(self):
        self.index = 0
        if self.shuffle == True:
            random.seed(20)
            random.shuffle(self.s) 
            random.seed(20)
            random.shuffle(self.labels)

    def next(self):
        if (self.is_available()):
            self.index += 1
            return self.data[self.index - 1]
        else:
            return

    def next_batch(self, batch_size):
        batch_size = min(self.data_size - self.index, batch_size)
        s_mats = []

        for i in range(batch_size):
            s = self.s[self.index + i]

            s_mats.append(np.expand_dims(np.pad(np.column_stack([self.word2vec.get(w) for w in s]),
                                                 [[0, 0], [0, self.max_len - len(s)]],
                                                 "constant"), axis=0))

        batch_s = np.concatenate(s_mats, axis=0)
        batch_s = batch_s.transpose((0, 2, 1))
        batch_labels = self.labels[self.index:self.index + batch_size]

        self.index += batch_size

        return batch_s, batch_labels



class WebQSP(Data):
    def open_file(self, pos_file, neg_file):
        pos = codecs.open(pos_file, encoding="utf-8").readlines()
        neg = codecs.open(neg_file, encoding="utf-8").readlines()
        pos = [clean_str(x).split(" ") for x in pos]
        neg = [clean_str(x).split(" ") for x in neg]
        self.s = np.array(pos + neg)
        pos_labels = [[0, 1] for _ in pos]
        neg_labels = [[1, 0] for _ in neg]
        self.labels = np.concatenate([pos_labels, neg_labels], 0)
        random.seed(10)
        random.shuffle(self.s) 
        random.seed(10)
        random.shuffle(self.labels)

        local_max_len = max([len(x) for x in s])
        if local_max_len > self.max_len:
            self.max_len = local_max_len

        self.data_size = len(self.s)


    def open_file_final(self, x_file, y_file):
        xs = codecs.open(x_file, encoding="utf-8").readlines()
        ys = codecs.open(y_file, encoding="utf-8").readlines()
        xs = [clean_str(x).split(" ") for x in xs]
        ys = [int(x.strip()) for x in ys]
        self.s = np.array(xs)
        new_y = []
        for y in ys:
            if y == 1:
                new_y.append([0, 1])
            elif y == 0:
                new_y.append([1, 0])
        self.labels = np.array(new_y)

        local_max_len = max([len(x) for x in self.s])
        if local_max_len > self.max_len:
            self.max_len = local_max_len

        self.data_size = len(self.s)





class SQ(Data):
    def open_file(self, pos_file, neg_file):
        pos = codecs.open(pos_file, encoding="utf-8").readlines()
        neg = codecs.open(neg_file, encoding="utf-8").readlines()
        pos = [clean_str(x).split(" ") for x in pos]
        neg = [clean_str(x).split(" ") for x in neg]
        self.s = np.array(pos + neg)
        pos_labels = [[0, 1] for _ in pos]
        neg_labels = [[1, 0] for _ in neg]
        #pos_labels = [1 for _ in pos]
        #neg_labels = [0 for _ in neg]
        self.labels = np.concatenate([pos_labels, neg_labels], 0)
        random.seed(10)
        random.shuffle(self.s) 
        random.seed(10)
        random.shuffle(self.labels)

        local_max_len = max([len(x) for x in self.s])
        if local_max_len > self.max_len:
            self.max_len = local_max_len

        self.data_size = len(self.s)




    def open_file_final(self, x_file, y_file):
        xs = codecs.open(x_file, encoding="utf-8").readlines()
        ys = codecs.open(y_file, encoding="utf-8").readlines()
        xs = [clean_str(x).split(" ") for x in xs]
        ys = [int(x.strip()) for x in ys]
        self.s = np.array(xs)
        new_y = []
        for y in ys:
            if y == 1:
                new_y.append([0, 1])
            elif y == 0:
                new_y.append([1, 0])
        self.labels = np.array(new_y)

        local_max_len = max([len(x) for x in self.s])
        if local_max_len > self.max_len:
            self.max_len = local_max_len

        self.data_size = len(self.s)




import time
import helper
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from BILSTM_CRF import BILSTM_CRF
import os

parser = argparse.ArgumentParser()
parser.add_argument("train_file", help="train file")
parser.add_argument("save_file", help="saved model")
parser.add_argument("-v","--val_file", help="validation file", default=None)
parser.add_argument("-e","--epoch", help="the number of epoch", default=100, type=int)
parser.add_argument("-c","--char_emb", help="the char embedding file", default=None)
parser.add_argument("-g","--gpu", help="the id of gpu, the default is 0", default=0, type=int)

args = parser.parse_args()

train_path = os.path.join(os.path.curdir, "data", args.train_file)
save_path = os.path.join(os.path.curdir, "model", args.save_file)
val_path = os.path.join(os.path.curdir, "data", args.val_file)
num_epochs = args.epoch
emb_path = args.char_emb
gpu_config = "/gpu:"+str(args.gpu)
num_steps = 21 # consist with the test

start_time = time.time()
print ("preparing train and validation data")
X_train, y_train, X_val, y_val = helper.getTrain(train_path=train_path, val_path=val_path, seq_max_len=num_steps)
char2id, id2char = helper.loadMap("char2id")
label2id, id2label = helper.loadMap("label2id")
num_chars = len(id2char.keys())
num_classes = len(id2label.keys())
if emb_path != None:
    embedding_matrix = helper.getEmbedding(emb_path)
else:
    embedding_matrix = None

print ("building model")
config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:
    with tf.device(gpu_config):
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            model = BILSTM_CRF(num_chars=num_chars, num_classes=num_classes, num_steps=num_steps, num_epochs=num_epochs, embedding_matrix=embedding_matrix, is_training=True)
        init = tf.global_variables_initializer()
        sess.run(init)
        print ("training model")
        model.train(sess, save_path, X_train, y_train, X_val, y_val)

        print ("final best f1 is: {:f}".format(model.max_f1))

        end_time = time.time()
        print ("time used {:f}(hour)".format((end_time - start_time) / 3600))

# -*- encoding: utf-8 -*-
from __future__ import division
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from tensorflow.contrib import learn
import csv
#from sklearn.metrics.pairwise import cosine_similarity
from preprocess import MData, Word2Vec
from datetime import datetime

# Data Parameters
tf.flags.DEFINE_string("final_x", "./data/webqspdata/WebQSP.final_test_x.landrprocessed.sost", "Final x")
tf.flags.DEFINE_string("final_y", "./data/webqspdata/WebQSP.final_test_y.txt", "Final y")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

x_raw = list(open(FLAGS.final_x, "r").readlines())
x_raw = [s.strip() for s in x_raw]
y_test = list(open(FLAGS.final_y, "r").readlines())    
y_test = [s.strip() for s in y_test]
y_test = list(map(int, y_test))
x_raw = [data_helpers.clean_str(sent) for sent in x_raw]
w = Word2Vec()
test_data = MData(word2vec=w)
test_data.open_file_final(FLAGS.final_x, FLAGS.final_y) 


print("\nEvaluating...\n")

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    session_conf.gpu_options.allow_growth=True
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
    
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
    
        #predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        scores = graph.get_operation_by_name("output/scores").outputs[0]
        
        all_scores = np.zeros((len(test_data.s), 2))           
        i = 0
        while test_data.is_available():
            x_test_batch, _ = test_data.next_batch(batch_size=FLAGS.batch_size)
            batch_scores = sess.run(scores, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_scores[i: i + len(x_test_batch)] = batch_scores
            i += len(x_test_batch)

     
if y_test is not None:
    processed_scores = [x[1] - x[0] for x in all_scores]
    #processed_scores = [x[1] for x in all_scores]
    preds = []
    labels = []
    tmp_y = []
    tmp_s = []
    tmp_y.append(y_test[0])
    tmp_s.append(processed_scores[0])
    correct = 0
    neg_sample = []
    pos_sample = []
    new_pos_file = open(os.path.join(FLAGS.checkpoint_dir, "..", 'WebQSP.filtered_pos_sample.txt'), mode='w')
    new_neg_file = open(os.path.join(FLAGS.checkpoint_dir, "..", 'WebQSP.filtered_neg_sample.txt'), mode='w')
    for i in range(1, len(y_test)):
        if y_test[i] == 1 and y_test[i - 1] == 0:
            labels.append(tmp_y)
            preds.append(tmp_s)
            pre_ans = dict(enumerate(tmp_s))
            find_true = 0
            pre_ans = sorted(pre_ans.items(), key = lambda item: item[1], reverse=True)
            pre_ans = pre_ans[:1]
            for p in pre_ans:
                if (tmp_y[p[0]] == 1):
                    find_true = 1
                else:
                    pos_sample.append(x_raw[i - len(tmp_y) + tmp_y.index(1)])
                    neg_sample.append(x_raw[i - len(tmp_y) + p[0]])
            if find_true == 1:
                correct += 1
            #else:
            #    print(len(labels))
            #    print(x_raw[i])
            #    print('result:')
            #    print(tmp_s)
            #    print('chosen:')
            #    print(pre_ans)
            tmp_y = []
            tmp_s = []
            tmp_y.append(y_test[i])
            tmp_s.append(processed_scores[i])
        else:
            tmp_y.append(y_test[i])
            tmp_s.append(processed_scores[i])
    labels.append(tmp_y)
    preds.append(tmp_s)
    for i in range(len(pos_sample)):
        new_pos_file.write(pos_sample[i] + '\n')
        new_neg_file.write(neg_sample[i] + '\n')
    total = len(preds)
    accu = correct/float(total)
    print("Total number of test examples: {}".format(total))
    print("Total number of correct predictions: {}".format(correct))
    print("Accuracy: {:g}".format(accu))

predictions_human_readable = np.column_stack((np.array(x_raw), processed_scores))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", datetime.now().strftime("%Y%m%d_%H%M%S") + "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)

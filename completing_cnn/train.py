# -*- encoding:utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import time
import datetime
from text_cnn import TextCNN
from tensorflow.contrib import learn
from preprocess import Word2Vec, MData

#tf.flags.DEFINE_string("positive_data_file", "./data/sqdata/SQ.t_and_v.pos.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("positive_data_file", "./data/webqspdata/WebQSPall.pos1.txt", "Data source for the positive data.")
#tf.flags.DEFINE_string("negative_data_file", "./data/sqdata/SQ.t_and_v.neg.txt", "Data source for the negative data.")
tf.flags.DEFINE_string("negative_data_file", "./data/webqspdata/WebQSPall.neg1.txt", "Data source for the negative data.")

tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding")
tf.flags.DEFINE_string("filter_sizes", "2,3,4,5", "Comma-separated filter sizes")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size")
tf.flags.DEFINE_float("dropout_keep_prob", 0.95, "Dropout keep probability")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda")

tf.flags.DEFINE_integer("batch_size", 64, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs")
tf.flags.DEFINE_integer("evaluate_every", 1, "Evaluate model on dev set after this many steps")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps")
tf.flags.DEFINE_integer("num_checkpoints", 1000, "Number of checkpoints to store")
#tf.flags.DEFINE_string("checkpoint_dir", "", "Chechpoint directory")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


print("Loading data...")
w = Word2Vec()
train_data = MData(word2vec=w)
train_data.open_file(FLAGS.positive_data_file, FLAGS.negative_data_file)
print("=" * 50)
print("training data size:", train_data.data_size)
print("training max len:", train_data.max_len)
print("=" * 50)


with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    session_conf.gpu_options.allow_growth=True
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=train_data.max_len,
            num_classes=2,
            #vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exist, so create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        #saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
        saver = tf.train.Saver(tf.global_variables())

        # Write vocabulary
        #vocab_processor.save(os.path.join(out_dir, "vocab"))

        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)


        for e in range(FLAGS.num_epochs):
            train_data.reset_index()
            while train_data.is_available():
                x_batch, y_batch = train_data.next_batch(batch_size=FLAGS.batch_size)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

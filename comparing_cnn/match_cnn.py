# -*- encoding:utf-8 -*-
import tensorflow as tf
import numpy as np

class MatchCNN():
    def __init__(self, s, w, l2_reg, num_features, d0=300, di=50, num_classes=2, num_layers=2):
        self.x1 = tf.placeholder(tf.float32, shape=[None, d0, s], name="x1")
        self.x2 = tf.placeholder(tf.float32, shape=[None, d0, s], name="x2")
        self.y = tf.placeholder(tf.int32, shape=[None], name="y")
        self.features = tf.placeholder(tf.float32, shape=[None, num_features], name="features")
	
        def pad_for_wide_conv(x):
            return tf.pad(x, np.array([[0, 0], [0, 0], [w - 1, w - 1], [0, 0]]), "CONSTANT", name="pad_wide_conv")

        def cos_sim(v1, v2):
            norm1 = tf.sqrt(tf.reduce_sum(tf.square(v1), axis=1))
            norm2 = tf.sqrt(tf.reduce_sum(tf.square(v2), axis=1))
            dot_products = tf.reduce_sum(v1 * v2, axis=1, name="cos_sim")

            return dot_products / (norm1 * norm2)

        def attention_matrix(x1, x2):
            euclidean = tf.sqrt(tf.reduce_sum(tf.square(x1 - tf.matrix_transpose(x2)), axis=1))
            return 1 / (1 + euclidean)

        def convolution(name_scope, x, d, reuse):
            with tf.name_scope(name_scope + "-conv"):
                with tf.variable_scope(name_scope + "conv") as scope:
                    conv = tf.contrib.layers.conv2d(
                        inputs=x,
                        num_outputs=di,
                        kernel_size=(d, w),
                        stride=1,
                        padding="VALID",
                        activation_fn=tf.nn.tanh,
                        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg),
                        biases_initializer=tf.constant_initializer(1e-04),
                        reuse=reuse,
                        trainable=True,
                        scope=scope
                    )
                    conv_trans = tf.transpose(conv, [0, 3, 2, 1], name="conv_trans")
                    return conv_trans


        def all_pool(variable_scope, x):
            with tf.variable_scope(variable_scope + "-all_pool"):
                if variable_scope.startswith("input"):
                    pool_width = s
                    d = d0
                else:
                    pool_width = s + w - 1
                    d = di

                all_ap = tf.layers.average_pooling2d(
                    inputs=x,
                    pool_size=(1, pool_width),
                    strides=1,
                    padding="VALID",
                    name="all_ap"
                )
                all_ap_reshaped = tf.reshape(all_ap, [-1, d])
                return all_ap_reshaped

        def CNN_layer(variable_scope, x1, x2, d):
            with tf.variable_scope(variable_scope):
                left_conv = convolution(name_scope="left", x=pad_for_wide_conv(x1), d=d, reuse=False)
                right_conv = convolution(name_scope="right", x=pad_for_wide_conv(x2), d=d, reuse=False)
                left_attention, right_attention = None, None

                att_mat = attention_matrix(left_conv, right_conv)
                left_attention, right_attention = tf.reduce_sum(att_mat, axis=2), tf.reduce_sum(att_mat, axis=1)
                    
                left_ap = all_pool(variable_scope="left", x=left_conv)
                right_ap = all_pool(variable_scope="right", x=right_conv)
                return left_ap, right_ap

        x1_expanded = tf.expand_dims(self.x1, -1)
        x2_expanded = tf.expand_dims(self.x2, -1)

        ques_raw = all_pool(variable_scope="input-left", x=x1_expanded)
        rel_raw = all_pool(variable_scope="input-right", x=x2_expanded)
        ques_processed, rel_processed = CNN_layer(variable_scope="CNN-1", x1=x1_expanded, x2=x2_expanded, d=d0)
        sims = [cos_sim(ques_raw, rel_raw), cos_sim(ques_processed, rel_processed)]


        with tf.variable_scope("output-layer"):
            self.output_features = tf.concat([self.features, tf.stack(sims, axis=1)], axis=1, name="output_features")
            self.estimation = tf.contrib.layers.fully_connected(
                inputs=self.output_features,
                num_outputs=num_classes,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg),
                biases_initializer=tf.constant_initializer(1e-04),
                scope="FC"
            )

        self.prediction = tf.contrib.layers.softmax(self.estimation)[:, 1]

        self.cost = tf.add(
            tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.estimation, labels=self.y)),
            tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)),
            name="cost")

        tf.summary.scalar("cost", self.cost)
        self.merged = tf.summary.merge_all()

        print("=" * 50)
        print("List of Variables:")
        for v in tf.trainable_variables():
            print(v.name)
        print("=" * 50)

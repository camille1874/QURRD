# -*- encoding:utf-8 -*-
from __future__ import division
import tensorflow as tf
import sys
import numpy as np

from preprocess import Word2Vec, WebQSP, SQ
from match_cnn import MatchCNN
from sklearn.externals import joblib
from helpers import build_path
import codecs




def test(w, l2_reg, epoch, max_len, data_type, classifier, word2vec, num_classes=2):
    if data_type == "WebQSP":
        test_data = WebQSP(word2vec=word2vec, max_len=max_len)
    else:
        test_data = SQ(word2vec=word2vec, max_len=max_len)

    test_data.open_file(mode="test")

    model = MatchCNN(s=max_len, w=w, l2_reg=l2_reg, num_features=test_data.num_features, num_classes=num_classes)
    model_path = build_path("./models/", data_type)
    accus = [] 
    print("=" * 50)
    print("test data size:", test_data.data_size)


    for e in range(40, 41):
        test_data.reset_index()
        f = codecs.open("result_webqsp" + str(e), encoding="utf-8", mode="w")
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, model_path + "-" + str(e))
            print(model_path + "-" + str(e), "restored.")

            if classifier == "LR" or classifier == "SVM":
                clf_path = build_path("./models_tmp/", data_type, "-" + str(e) + "-" + classifier + ".pkl")
                clf = joblib.load(clf_path)
                print(clf_path, "restored.")

            QA_pairs = {}
            s1s, s2s, labels, features = test_data.next_batch(batch_size=test_data.data_size)
            for i in range(test_data.data_size):
                pred, estimation, clf_input = sess.run([model.prediction, model.estimation, model.output_features], feed_dict={model.x1: np.expand_dims(s1s[i], axis=0), model.x2: np.expand_dims(s2s[i], axis=0), model.y: np.expand_dims(labels[i], axis=0), model.features:np.expand_dims(features[i], axis=0)})

                if classifier == "LR":
                    clf_pred = clf.predict_proba(clf_input)
                    pred = clf_pred[0][1] - clf_pred[0][0]
                elif classifier == "SVM":
                    clf_pred = clf.decision_function(clf_input)
                    pred = clf_pred

                s1 = " ".join(test_data.s1s[i])
                s2 = " ".join(test_data.s2s[i])
	        
                f.write(s1 + "  " + s2 + " " + str(pred) + "\n")

                if s1 in QA_pairs:
                    if test_data.s1s[i - 1] == test_data.s1s[i]:
                        if s1 + "1" not in QA_pairs:
                            QA_pairs[s1].append((s2, labels[i], np.asscalar(    pred)))                 
                        else:
                            t = 1
                            while s1 + str(t) in QA_pairs:
                                t += 1
                            QA_pairs[s1 + str(t - 1)].append((s2, labels[i],     np.asscalar(pred)))
                    else:
                        t = 1
                        while s1 + str(t) in QA_pairs:
                            t += 1
                        QA_pairs[s1 + str(t)] = [(s2, labels[i], np.asscalar    (pred))]
                else:
                    QA_pairs[s1] = [(s2, labels[i], np.asscalar(pred))]
     


            accu = 0
            for s1 in QA_pairs.keys():

                QA_pairs[s1] = sorted(QA_pairs[s1], key=lambda x: x[-1], reverse=True)
                if QA_pairs[s1][0][1] == 1:
                    accu += 1

            num_questions = len(QA_pairs.keys())
            accu /= num_questions

            accus.append(accu)
            print("[Epoch " + str(e) + "] accu:", accu)

    print("=" * 50)
    print("max accu:", max(accus))
    print("=" * 50)

    exp_path = build_path("./experiments/", data_type, "-" + classifier + ".txt")
    r = "Epoch\taccu\n"
    f = codecs.open(exp_path, "w", encoding="utf-8")
    for i in range(len(accus)):
        r += str(39 + i) + "\t" + str(accus[i]) + "\n"
    f.write(r)

if __name__ == "__main__":
    # default parameters
    params = {
        "ws": 4,
        "l2_reg": 0.0004,
        "epoch": 50,
        "max_len": 28,
        "data_type": "WebQSP",
        "classifier": "model",
        "word2vec": Word2Vec()
    }

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            k = arg.split("=")[0][2:]
            v = arg.split("=")[1]
            params[k] = v

    test(w=int(params["ws"]), l2_reg=float(params["l2_reg"]), epoch=int(params["epoch"]),
         max_len=int(params["max_len"]), data_type=params["data_type"],
         classifier=params["classifier"], word2vec=params["word2vec"])

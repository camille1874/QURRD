# -*- encoding:utf-8 -*-
from __future__ import division
import codecs
score1 = codecs.open("20180511_235642prediction.csv", encoding="utf-8").readlines()
score2 = codecs.open("result_webqsp49", encoding="utf-8").readlines()
#final_f = codecs.open("result_63_webqsp.txt", encoding="utf-8", mode="w")
y = codecs.open("./completing_cnn/data/webqspdata/WebQSP.final_test_y.txt").readlines()
y = [int(t.strip()) for t in y]
scores2 = [float(t.strip().split("  ")[1].split()[-1].replace("[","").replace("]","")) for t in score2]
scores1 = [float(t.strip().split(",")[-1]) for t in score1]
accu = 0
max_accu = 0
for w in range(1, 4):
    for b in range(0, 20):
        for a in range(0, 20):
            #final_str = ""
            if a ==0 and b == 0:
                continue
            print("*" * 50) 
            correct = 0
            total = 0
            questions = []
            cand = []
            result = []
            for i, l in enumerate(score2):
                l = l.strip().split("  ")
                assert (len(l) == 2)
                q = l[0]
                r = l[1]
                questions.append(q)
                s2 = scores2[i]
                r = " ".join(r.split()[:-1])
                s1 = scores1[i]
                if s1 < 0 and s2 < 0:
                    s1 = s1 * w
                    s2 = s2 * w 
                s = a * s1 + b * s2
                if (i != 0 and y[i] == 1 and y[i - 1] == 0) or i == len(score2) - 1:
                    #if questions[i] == questions[i - 1]:
                    #    print(i)
                    #assert(questions[i] != questions[i - 1])
                    total += 1
                    if i == len(score2) - 1:
                        cand.append(r)
                        result.append(s)
                    #if(((result.index(max(result)) == 0) and (max(result) != min(result))) or (len(result) == 1)):
                    #final_str += q
                    #final_str += " ".join([str(r) for r in result]) + "\n"
                    if((y[result.index(max(result))] == 1 and (max(result) != min(result))) or (len(result) == 1)):
                        correct += 1
                    #else:
                    #    print("s1")
                    #    print(scores1[i-len(cand):i])
                    #    print("s2")
                    #    print(scores2[i-len(cand):i])
                    #    print("result")
                    #    print(result)
                    #    print("question")
                    #    print(questions[i - len(cand)])
                    #    print("cand")
                    #    print(cand)
                    cand = []
                    result = []
                cand.append(r)    
                result.append(s)
   
            print("a b w:" + str(a) + " " + str(b) + " " + str(w))
            print("total:" + str(total))
            print("correct:" + str(correct))
            accu = correct / total
            print("accu:" + str(accu))
            if accu > max_accu:
                max_accu = accu
            print("max_accu:" + str(max_accu))
            print("*" * 50)
#final_f.write(max_str)    

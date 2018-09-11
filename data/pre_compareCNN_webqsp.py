# -*- encoding:utf-8 -*-
import codecs
import random
source = codecs.open('./train/train.WebQSP.replace_ne.txt', encoding='utf-8').readlines()
#labels = codecs.open('WebQSP.label_result.train.out.processed', encoding='utf-8').readlines()
relations = codecs.open('../KG/WebQSP_relation.processed', encoding='utf-8').readlines()
target_x = codecs.open('WebQSPall-train-raw.txt', mode='w', encoding='utf-8')
#target_y = codecs.open('WebQSP.final_test_y.txt', mode='w', encoding='utf-8')
#target_x1 = codecs.open('WebQSPall.pos.txt', mode='w', encoding='utf-8')
#target_x2 = codecs.open('WebQSPall.neg.txt', mode='w', encoding='utf-8')
for i in range(len(source)):
    line = source[i]
    line = line.strip().split('\t')
    line[0] = line[0].split()
    line[1] = line[1].split()
    question = line[-1].replace("<e>", "entity").replace('$ARG1', '').replace('$ARG2', '').strip()
    #pos_count = len(line[0])
    #for j in range(pos_count):
    #    pos = int(line[0][j].strip())
    #    neg = int(line[1][j].strip())
    #    pos_file.write(question + ' ' + labels[i].strip() + ' ' + relations[pos - 1].strip() + '\n')
    #    neg_file.write(question + ' ' + labels[i].strip() + ' ' + relations[neg - 1].strip() + '\n')
    pos = list(map(int, line[0]))
    neg = list(map(int, line[1]))
    for p in pos:
        if p in neg:
            neg.remove(p)
        #target_x.write(question + ' ' + labels[i].strip().replace("someone", "someone something") + '\t' + relations[p - 1].strip() + '\t' + str(1) + '\n')
        target_x.write(question + '\t' + relations[p - 1].strip() + '\t' + str(1) + '\n')
        #target_x1.write(question + ' ' + labels[i].strip().replace("someone", "someone something") + "  " + relations[p - 1].strip() + "\n")
    #for n in neg[:max(10, len(neg) / 8)]:
    random.shuffle(neg)
    #for n in neg[:60]:
    for n in neg:
        #target_x.write(question + ' ' + labels[i].strip().replace("someone", "someone something") + '\t' + relations[n - 1].strip() + '\t' + str(0) + '\n')
        target_x.write(question + '\t' + relations[n - 1].strip() + '\t' + str(0) + '\n')
        #target_x2.write(question + ' ' + labels[i].strip().replace("someone", "someone something") + "  " + relations[n - 1].strip() + "\n")

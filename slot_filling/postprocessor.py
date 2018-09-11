# -*- encoding:utf-8 -*-
import codecs
verbs = codecs.open('./data/dict/en_verb_inflections.txt', encoding='utf-8').readlines()
source = codecs.open('./result/SQ.valid.sf', encoding='utf-8').readlines()
target = codecs.open('./result/SQ.valid.sf.processed', encoding='utf-8', mode='w')
verb_dict = dict()
verb_all = []
for line in verbs:
    line = line.strip().replace('\'', '').split('\t')
    verb_all.append(line)
for v in verb_all:
    for j in v[1:]:
        verb_dict[j] = v[0]
for line in source:
    #if line == '\n':
    #    target.write(line)
    #    continue
    #l = line.strip().split('\t')
    l = line.strip().split(' ')
    #raw_verb = l[0]
    #pad = l[1]
    result = ''
    for raw_verb in l:
        raw_verb = raw_verb.strip()
        if raw_verb in verb_dict:
            re_verb = verb_dict[raw_verb]
        elif raw_verb in ['is', 'are', 'was', 'were']:
            re_verb = 'be'
        elif raw_verb in ['isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t']:
            print raw_verb
            re_verb = 'be not'
        elif raw_verb in ['has', 'had']:
            re_verb = 'have'
        elif raw_verb in ['hasn\'t', 'haven\'t', 'hadn\'t']:
            re_verb = 'have not'
        elif raw_verb == 'where':
            re_verb = 'location place'
        elif raw_verb == 'who':
            re_verb = 'people person'
        elif raw_verb == 'when':
            re_verb = 'time date'
        else:
            re_verb = raw_verb
        result = result + re_verb + ' '  
    #target.write(re_verb + '\t' + pad + '\n')
    target.write(result + '\n')


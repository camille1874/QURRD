#!/bin/bash

echo "Downloading SimpleQuestions and WebQSP relation detection dataset...\n"
git clone https://github.com/Gorov/KBQA\_RE\_data
mv KBQA_RE_data/sq_relations/train.replace_ne.withpool ./data/train/
mv KBQA_RE_data/sq_relations/valid.replace_ne.withpool ./data/valid/
mv KBQA_RE_data/sq_relations/test.replace_ne.withpool ./data/test/
mv KBQA_RE_data/webqsp_relations/WebQSP.RE.train.with_boundary.withpool.dlnlp.txt data/train/train.WebQSP.replace_ne.txt
mv KBQA_RE_data/webqsp_relations/WebQSP.RE.test.with_boundary.withpool.dlnlp.txt data/test/test.WebQSP.replace_ne.txt



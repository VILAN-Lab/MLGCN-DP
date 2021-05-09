from stanfordcorenlp import StanfordCoreNLP
import json
from tqdm import tqdm
path = './stanford-corenlp-4.2.1'
nlp = StanfordCoreNLP(path)
Data = ['train', 'val', 'test']
for data in Data:
    sent1 = []
    sent2 = []
    sent3 = []
    sent4 = []
    with open('./data/{}/{}.post'.format(data, data)) as f:
        count = 0
        s_max = 0
        m = 0
        for line in tqdm(f):
            line = line.strip().split('\t')
            s1 = line[0].replace("'", '').replace(' .', '')
            s2 = line[1].replace("'", '').replace(' .', '')
            s3 = line[2].replace("'", '').replace(' .', '')
            s4 = line[3].replace("'", '').replace(' .', '')
            s1 = [nlp.dependency_parse(s1)]
            s2 = [nlp.dependency_parse(s2)]
            s3 = [nlp.dependency_parse(s3)]
            s4 = [nlp.dependency_parse(s4)]
            sent1.append(s1)
            sent2.append(s2)
            sent3.append(s3)
            sent4.append(s4)
        relation = {'sent1': sent1,
                    'sent2': sent2,
                    'sent3': sent3,
                    'sent4': sent4,
        }
    json.dump(relation, open('dependency_relation_%s.json' % data, 'w'))
nlp.close()
print('all done!')
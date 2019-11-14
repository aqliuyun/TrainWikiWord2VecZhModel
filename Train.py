import warnings
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')
import os.path
import sys
import jieba
import re
import multiprocessing
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from opencc import OpenCC

if __name__ == '__main__':
    multiprocessing.freeze_support()
    raw_file = open('./models/wiki.zh.txt', 'w', encoding='utf-8')
    wiki = WikiCorpus('./zhwiki-latest-pages-articles.xml.bz2', lemmatize=False, dictionary={})
    cn_reg = '^[\u4e00-\u9fa5]+$'
    cc = OpenCC('t2s')
    i = 0
    for text in wiki.get_texts():             
        line_list_new = []
        for word in text:
            word = word.replace('\n','').replace(' ','')                     
            word = cc.convert(word)
            if re.search(cn_reg, word):
                line_list_new.append(word)
        line = ''.join(line_list_new)
        line = ' '.join(jieba.cut(line))
        raw_file.write(line + "\n")
        i = i + 1
        if (i % 1000 == 0):
            print("Saved " + str(i) + " articles")
    raw_file.close()
    print('contine?')
    wait = Input()    
    model = Word2Vec(LineSentence('./models/wiki.zh.txt'), size=400, window=5, min_count=5,
            workers=multiprocessing.cpu_count())
    model.init_sims(replace=True)
    model.save('./models/wiki.zh.model')
    model.save_word2vec_format('./models/wiki.zh.bin', binary=True)

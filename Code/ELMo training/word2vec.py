import os
import codecs
from gensim.models import word2vec
import nltk
import re
import gensim
from sklearn.decomposition import PCA
from matplotlib import pyplot
def cleanReview(txt,w_path):
    sents =nltk.sent_tokenize(txt)
    i=0
    list =[]
    for sent in sents:
        m=re.search("Fig.",sent)
        if m!=None:
            list.append(i)
        i = i + 1
    b=0
    for i in list:
        sents[i-b]=sents[i-b]+sents[i+1-b]
        del sents[i+1-b]
        b = b + 1
    for sent in sents:
        words = nltk.word_tokenize(sent)
        word = []
        for x in words:
            if x!=":" and x!="." and x!=",":
                word.append(x)
        write(word,w_path)

def write(word,w_path):
    with codecs.open(w_path,"a",encoding="utf-8") as f:
        for x in word:
            f.write("%s " % x)
        f.write("\n")

def from_ann2dic(r_txt_path,w_path):
    with codecs.open(r_txt_path,"r",encoding="utf-8") as f:
        content_str=f.read()
        cleanReview(content_str,w_path)

if __name__=="__main__":

    data_dir = "project"
    for file in os.listdir(data_dir):
        if file.find(".")==-1:
            continue
        file_name =file[0:file.find(".")]
        if len(file_name)==0:
            continue
        r_txt_path = os.path.join(data_dir,"%s.txt" % file_name)
        w_path = "new.txt"
        from_ann2dic(r_txt_path,w_path)
        sentences = word2vec.LineSentence(w_path)
        model = gensim.models.Word2Vec(sentences, size=32, sg=1, iter=8)
        model.wv.save_word2vec_format("./B.vecs.lc.over100freq" + ".txt", binary=False)
        model = gensim.models.KeyedVectors.load_word2vec_format("vecs.lc.over100freq.txt", binary=False)
        
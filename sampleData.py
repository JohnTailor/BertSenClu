from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
import numpy as np

#Preprocess 20News dataset (some operations, replacemnts are dataset specific, i.e., there are many single chars, tabs and ">" in the docs, also some docs are very short)
#Input: docs...documents, labels...categories of documents (optional), minLen...docs with length below minlen get filtered, nsamp... number of documents to return (-1 all)
#Output: preprocessed docs (potentially less than original ones), labels of preprocessed docs
def preprocess20News(docs,labels=None,minLen=40,nsamp=1):
    nx, ny = [], []
    stopw = stopwords.words('english')
    for i,x in enumerate(docs):
        cx = x.split("\n")
        cx = " ".join([s for s in cx if len(s) > 1])
        cx = cx.replace("\t", " ").replace(">", " ").replace(":", " ").replace("   ", " ").replace("   ", " ").replace("  ", " ")
        ws = cx.split(" ")
        ws = [w for w in ws if len(w) > 1 or w.lower() in stopw]  # remove single letters (quite common in this dataset), except if it is a letter that is meaningful, e.g., like "I" or an article like "a"
        cx = " ".join(ws)
        if len(cx) < minLen: continue  # remove too short docs
        nx.append(cx)
        if not labels is None: ny.append(labels)
        if nsamp > 0 and len(nx) >= nsamp: break
    print("Pre-processing done: ", np.round(len(nx)/len(docs)*100), "[%] of documents removed because length < min threshold ",minLen)
    return nx, None if labels is None else ny

#Preprocessing of 20newsgroups (this was applied for all topic models (they potentially did additional preprocessing)
def get20NewsData(nsamp=-1):
    print("Get data")
    dstrain, dstest = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes')), fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
    datx = dstrain.data + dstest.data
    daty = np.concatenate((dstrain.target, dstest.target))
    nx,ny=preprocess20News(datx,daty,nsamp=nsamp)
    print("Total documents", len(nx))
    return nx,ny


def getNLTK(nsamp=-1,data="Reuters"):
    print("Get data")
    from nltk.corpus import reuters,movie_reviews
    corpus=reuters if data=="Reuters" else movie_reviews
    fileids = corpus.fileids()
    nx=[]
    for file_id in fileids:
        file_words = corpus.words(file_id)
        output = " ".join(file_words)
        output=output.replace(" ;",";").replace(" :",":").replace(" .",".").replace(" ' ","'").replace(" ,",",").replace('. "','."')
        if len(str(output))>10: #filter none etc.
            nx.append(output)
        if len(nx)==nsamp: break
    print("Total documents", len(nx))
    return nx,None
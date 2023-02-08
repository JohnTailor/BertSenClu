#Python libs
import numpy as np,os,pickle
import multiprocessing
from multiprocessing import Pool
import sys
sys.path.append('bertSenClu') #for visutils import
import visUtils

#PyTorch
import pandas as pd
import torch
import torch.cuda.amp as tca
import torch.nn as nn

#Other libs
from gensim import utils as gut #tokenizer
from sentence_transformers import SentenceTransformer #Sentence embedder see https://www.sbert.net/; pip install -U sentence-transformers
import pysbd #Sentence segmenter https://github.com/nipunsadvilkar/pySBD #pip install pysbd
from nltk.stem import WordNetLemmatizer

class SenClu():
    def __init__(self,device='auto'):
        """
        Inputs:
        device...where to do computations : 'cpu' or 'cuda' (= on GPU) or 'auto' (= choose cuda if available else cpu)
        """
        super(SenClu, self).__init__()
        self.maxTopWords = 50 #Top words per Topic computed
        self.device=device
        self.verbose=True


    def computeTopicModel_SenClu(self,docs, device, ntopics, nepoch=20, alpha=1,epsilon=0.001): #dimension of sentence embeddings # initial offset to add to topic to smoothen p(t|d) distribution; # alpha = final offset, serves also as prior to determine how many topics per document exist
        """
        Compute topic model given docs with sentence embeddings
        ---------------------------------------------
        Inputs:
        docs = documents as string
        device = cuda (GPU) or cpu
        ntopics = number of topics
        nepoch = number of training epochs
        alpha = parameter giving the preference to have many or few topics per document, i.e., we used p(w|topic t) = p(w,vec topic t) *(alpha+ probability(topic t |document) , 0...means tend towards having only 1 topic, alpha...probability of topic in document is proportional to alpha even if no sentence of document belongs to topic t
        epsilon = Convergence criterion: Stop computation if topic distribution changes by less than this for an epoch (even before nepoch have been executed)
        ----------------------
        Returns:
        ptd... probability topic given document (for each doc in input) ;Dimensions: #topics x #docs
        vec_t... topic vectors; Dimensions: #topics x embedding dim of pretrained vecs:
        assign_t ... topic assignments to each sentences;  ~ #docs x #sentences in doc
        """
        if self.verbose: print("Training topic model")
        nd = len(docs)  # number of docs
        istart = ioff= 8 #initial smoothening of p(t|d), not very critical, if doc is very long, it should also be a bit larger
        nWarmUp=0 #number of warm Up epochs -> cluster more documents and later split them; (not needed)

        # Initialize topic vectors v_t,
        embDims=len(docs[0][0]) # embDims = dimensions of pre-trained sentence encodings, usually 384
        vec_t = (np.random.random((embDims, ntopics)) - 0.5)  # Initialize topic vectors randomly
        sum_vt = np.sqrt(np.sum(vec_t ** 2, axis=0, keepdims=True))
        vec_t = vec_t / (sum_vt + 1e-10)
        vec_t = vec_t.astype(dtype=np.float32)

        closs = 1  # loss
        assignSen_ptd = [[] for _ in range(nd)]  # topic assignments of each sentence account for prob(t|d)
        #assignSen = [[] for _ in range(nd)]  # topic assignments of each sentence without prob(t|d)
        prob_t = [[] for _ in range(nd)]  # topic prob of assigned sentence
        with tca.autocast():
            ptd = torch.ones((ntopics, nd)) / ntopics  # p(t,d) uniform
            ptd = ptd.to(device)
            vec_t = torch.from_numpy(vec_t)
            vec_t = vec_t.to(device)
            docs_torch = [torch.from_numpy(x.astype(np.float32)) for x in docs]
            docs_torch = [x.to(device) for x in docs_torch]

        #Start training
        for epoch in range(nepoch):

            cbatches = np.random.permutation(len(docs_torch))  # we don't use the dataloader but rather permute the batch ids
            nAssign_t = torch.zeros(ntopics).to(device)  # number of assigned sentences to topic
            newvec_t = torch.zeros_like(vec_t).to(device)  # This will become the new topic vector v_t after the epoch

            for ib, i in enumerate(cbatches):  # go through all docs
                dx = docs_torch[i]  # current document
                with tca.autocast():
                    with torch.no_grad():
                        # Compute similarity of document, i.e., its sentences, and topic vectors
                        pts = torch.matmul(dx, vec_t)  # Dot product of sentence vectors in doc and topic vectors
                        prod = pts * (ptd[:, i] + ioff)  # Compute p(s|t)*p(t|d) , (mean is not needed) #* torch.mean(pts[pts > 0]
                        vals, mind = torch.max(prod, dim=1)  # Compute most likely topic and take it as assignment

                        # Update p(t|d)
                        cntd = nn.functional.one_hot(mind, ntopics)
                        sntd = torch.sum(cntd, dim=0)
                        ptd[:, i] = sntd / torch.sum(sntd)

                        # Update topic vectors
                        td = torch.transpose(dx, 0, 1)
                        prod = torch.matmul(td, cntd.float())
                        sntd = sntd * ptd[:, i]
                        prod = prod * ptd[:, i]
                        newvec_t = newvec_t + prod  # use only some of sentence vectors -> could weight by ptd and do rolling update
                        nAssign_t += sntd

                        # Other stuff:
                        assignSen_ptd[i] = mind.cpu().numpy()  # Remember assignment (only needed at the end)
                        prob_t[i]=vals.cpu().numpy()
                        closs = 0.95 * closs + 0.05 * torch.mean(vals).item() if epoch>0 or i>10 else 0.8 * closs + 0.2 * torch.mean(vals).item() # update loss

            if self.verbose: print("  Epoch:",epoch," Loss",  np.round(closs, 4))
            if epoch>=nWarmUp: #warmup epochs, where we do more of a document clustering with small alpha; this helps to distribute frequent words
                ioff = istart if epoch==nWarmUp else max(ioff / 2, alpha)  # Update the smoothing constant
            else:
                ioff = alpha
            with torch.no_grad():  # Update the topic vector
                newvec_t = newvec_t / (nAssign_t + 1e-8)
                diff = torch.sqrt(torch.sum((newvec_t - vec_t) ** 2))
                vec_t = newvec_t
                if diff < epsilon and epoch>4: break # Stop if vectors don't change significantly anymore
        return ptd.cpu().numpy(), vec_t, assignSen_ptd,prob_t


    """
    Lemmatize topics
    -----------
    Inputs
    lemDict...Dictionary of word -> lemma
    topics... list of list of words resembling top words of topics
    Returns
    lemmatized topics... list of list of words resembling top lemmatized words of topics
    """
    def lemTopics(self,topics, lemDict):
        nt=[]
        for t in topics:
            ct=[]
            for w in t:
                cw=w
                if cw in lemDict:
                    cw=lemDict[w]
                elif cw.lower() in lemDict:
                    cw=lemDict[cw.lower()]
                if not cw in ct:
                    ct.append(cw)
            nt.append(ct)
        return nt

    def get_topics(self,sendocs, assign_t):
        """
        Compute topics (list of words)
        -------------------
        Input
        ptd... probability topic given document (for each doc in input) ;Dimensions: #topics x #docs
        vec_t... topic vectors; Dimensions: #topics x embedding dim of pretrained vecs:
        assign_t ... topic assignments to each sentences;  ~ #docs x #sentences in doc
        ntopw ... number of words to return per topic (words with largest score are returned)
        -------------------
        Return
        topics ... list of list of words; each list of words contains ntopw words
        """
        #Preprocess corpus -> We need to tokenize documents into words, we also lower case and lemmatize
        # NOTE: Other methods use their own preprocessing, we also lemmatize the final topics of other methods
        dic = {}  # dictionary word to id
        nw = 2000  # number of initial words (this is grown)
        nfw = 0
        nd = len(sendocs)
        nt = self.ntopics
        occ_wt = np.zeros((nw, nt), dtype=np.int32)  # matrix with number of occ of word in topic (it is grown as needed)
        occ_wd = np.zeros((nw, nd), dtype=np.int32)  # matrix with number of occ of word in document
        occ_dt = np.zeros((nd, nt), dtype=np.int32)  # matrix with number of occ of topic per document
        #from nltk.corpus import stopwords
        #stop_words = set(stopwords.words('english'))
        lemmatizer=WordNetLemmatizer()
        corpus = []
        lems={} #build map of words to lemmas
        for id in range(nd):  # go through all docs
            dass = assign_t[id]  # assignments of sentences of doc
            doc = sendocs[id]  # sentences in doc
            if self.verbose and len(dass) != len(doc):
                print("Failed not all sentences assigned", id, len(dass), len(doc)," NDoc", nd, len(sendocs),doc)
            cdoc = []
            for it, s in enumerate(doc):  # go through all sentences
                toks = gut.tokenize(s, lower=True)
                toks = [ctok for ctok in toks if len(ctok) > 1]  # ignore single chars
                #toks = [ctok for ctok in toks if not ctok in stop_words]  # Optional: Igore stop words
                for w in toks:
                    if not w in lems:
                        lems[w] = lemmatizer.lemmatize(w)
                toks = self.lemTopics([toks], lems)[0]
                ctopic = dass[it]
                for w in toks:  # go through all tokens
                    cdoc.append(w)
                    if not w in dic:  # add new word to dic
                        dic[w] = nfw
                        nfw += 1
                        if nfw > occ_wt.shape[0]:  # if found max number of words expand matrix
                            occ_wt = np.concatenate([occ_wt, np.zeros((nw, nt), dtype=np.int32)], axis=0)
                            occ_wd = np.concatenate([occ_wd, np.zeros((nw, nd), dtype=np.int32)], axis=0)
                    occ_wt[dic[w], ctopic] += 1  # add count of words in topic
                    occ_wd[dic[w], id] += 1
                    occ_dt[id, ctopic] += 1
            corpus.append(cdoc)

        denom = np.sum(occ_wt, axis=1)+1e-8
        denom=denom.reshape(-1, 1)
        #scores_word_topic = (occ_wt-denom/self.ntopics) ** 0.5 * np.clip(occ_wt / denom-1/self.ntopics,0,1) #This is an extenstion from the paper: Essentially, by look at how much more the probability is above "chance", the advantage is that otherwise we have strong dependency o nthe number of topics, i.e. , say we have ntopics=2 and the word "a" occurs 1000 times, by chance it occurs very frequent everywhere, byt for ntopics=200. by chance it occurs a factor 100 less # * np.sum(awt, axis=0,keepdims=True)/np.sum(awt)
        scores_word_topic = np.clip((occ_wt - denom / self.ntopics) ,0,1e10)** 0.5 * np.clip(occ_wt / denom - 1 / self.ntopics, 0,1)

        #Get indexes of words with top scores
        sw=np.zeros((self.maxTopWords,nt),dtype=np.int32)
        swscores = np.zeros((self.maxTopWords, nt), dtype=np.float32)
        index_top_scores=np.copy(scores_word_topic)
        for i in range(self.maxTopWords):
            wind=np.argmax(index_top_scores,axis=0)
            mscore=np.max(index_top_scores,axis=0)
            sw[i]=wind
            swscores[i]=mscore
            index_top_scores[wind,np.arange(nt)]=-1 #set to maximum so that wont be taken again

        #Get words per topic from indexes
        idic = {dic[w]: w for w in dic}
        topicsAsWordScoreList=[]
        for t in range(nt):
            ws=sw[:,t]
            scores=swscores[:,t]
            #lws=[idic[w] for w in ws if w in idic]
            lws = [(idic[w],s) for w,s in zip(ws,scores)]
            topicsAsWordScoreList.append(lws)
        return topicsAsWordScoreList


    def segmenter(self,docs):
        """
        Segment docs into sentences
        Inputs: list of documents, each document being a string
        Returns: list of documens, each document being a list of sentences
        """
        pid,docs=docs
        seg = pysbd.Segmenter(language="en", clean=False)
        adocs=[]
        out=300
        for id,d in enumerate(docs):
            segs = seg.segment(d)
            adocs.append(segs)
            if self.verbose and id==out:
                print("   ProcessID:",pid," Segmented by process:",id)
                out*=4
        return adocs


    #Split list a into n parts
    def split(self,a, n):
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


    def getEncodedSentences(self,docs,device):
        """
        Segment docs into sentences
        Input: list of documents, each document being a string
        Returns: list of documens, each document being a list of sentences
        """
        if self.verbose: print("Segmentation of docs into sentences; Total Docs: ",len(docs))
        #Segment documents into sentences
        nseg=min(8,multiprocessing.cpu_count()-1)
        px = list(self.split(docs, nseg))
        cpool = Pool(nseg)
        docs_segmented = cpool.map(self.segmenter, zip(np.arange(len(px)),px))
        docs_segmented = sum(docs_segmented, [])
        filtered_docs=[]
        nFailed=0
        for i,d in enumerate(docs_segmented):
            if len(d)==0:
                if self.verbose and nFailed<40: print("Sentence embedded doc is empty, removing it (e.g., only 'return characters' etc.); Original doc ID",i," Original doc length:",len(docs[i]), " Content (enclosed in --):  --"+docs[i]+"--")
                nFailed+=1
            else: filtered_docs.append(d)
        if nFailed>0: print(" Number of empty sentenced embedded docs:",nFailed, "  (Use verbose option to see docs")
        if self.verbose: print("Embed sentences using a sentence embedder; Total Docs: ",len(filtered_docs))
        #get Sentence Embeddings -> this can be parallelized
        model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        model.eval()
        docs_sentenceEncoded = []
        out = 300
        for id, d in enumerate(filtered_docs):
            with torch.no_grad():
                with tca.autocast():
                    segs = model.encode(d)
            docs_sentenceEncoded.append(segs)
            if self.verbose and id == out:
                print("  Encoded Docs:", id)
                out *= 4
        return filtered_docs,docs_sentenceEncoded


    def fit_transform(self, docs, nTopics=50, alpha=None, nEpochs=40, loadAndStoreInFolder=None, verbose=True):
        """
        Compute topic_model given raw text docs, optional: stores partial sentence embeddings for recomputation with other parameters
        --------------------
        Input:
        docs...documents
        alpha = parameter giving the preference to have many or few topics per document, i.e., we used p(w|topic t) = p(w,vec topic t) *(alpha+ probability(topic t |document) , 0...means tend towards having only 1 topic, alpha...probability of topic in document is proportional to alpha even if no sentence of document belongs to topic t
        ntopwords...number of words returned per topic (they are sorted in descending order by relevance)
        storeInFolder...path to a folder, where docs (split into sentences) and sentence embeddings of docs are stored; if they exist, these are loaded and used for topic modeling ; this speeds up comutation of different models on same corpus
        verbose...Print state of computation during topic modeling process
        Returns:
        topics...list of topics, each topic contains a list of "ntopwords" words sorted in descending order by relevance
        probs...topic probabilities
        """
        if alpha is None: alpha=1/np.sqrt(nTopics)
        if self.device=='auto':
            device='cuda' if torch.cuda.is_available() else 'cpu'
        self.verbose=verbose
        # get encoded Sentences
        if loadAndStoreInFolder is None or not os.path.exists(loadAndStoreInFolder + "topicDat.pic"):
            self.sendocs, embdocs = self.getEncodedSentences(docs, device)
            if not loadAndStoreInFolder is None:
                os.makedirs(loadAndStoreInFolder, exist_ok=True)
                with open(loadAndStoreInFolder + "/topicDat.pic", "wb") as f:
                    pickle.dump([self.sendocs, embdocs], f)
        else:
            if self.verbose: print("Loading docs split into sentences and embedded sentences for faster topic modeling")
            with open(loadAndStoreInFolder + "/topicDat.pic", "rb") as f:
                 self.sendocs, embdocs = pickle.load(f)


        self.ntopics=nTopics
        # Topic modeling and word-topic extraction
        self.ptd, self.vec_t, self.assignSen_ptd,self.prob_t=self.computeTopicModel_SenClu(embdocs, device, nTopics, nepoch=nEpochs, alpha=alpha)
        self.topicsAsWordScoreList = self.get_topics(self.sendocs, self.assignSen_ptd)
        self.pt=np.sum(self.ptd,axis=1)/self.ptd.shape[1]
        return self.getTopics(),self.pt

    def getTopics(self,wordsPerTopic=10): return [[w for w, s in t[:wordsPerTopic]] for t in self.topicsAsWordScoreList]

    def getTopicsWitchScores(self): return self.topicsAsWordScoreList

    def getTopicDocumentDistribution(self): return self.ptd

    def getTopicDistribution(self): return self.pt


    def getTopDocsPerTopic(self,ntop=20):
        sortDocs=np.argsort(self.ptd,axis=1) #ascending order!
        topDocs=[]
        for i in range(self.ntopics):
            topd=sortDocs[i][-ntop:]
            topd=topd[::-1] #descending order
            scores=self.ptd[i][topd]
            csen=[self.sendocs[itop] for itop in topd]
            topDocs.append(list(zip(csen,scores,topd)))
        return topDocs #return document ID, doc score and doc


    def getTopSentencesPerTopic(self, ntop=20):
        perTop=[[] for _ in range(self.ntopics)]
        for i in range(len(self.sendocs)):
            vals=self.prob_t[i]
            tops=self.assignSen_ptd[i]
            for j in range(len(vals)):
                perTop[tops[j]].append((vals[j],i,j))
        for t in range(self.ntopics):
            s=sorted(perTop[t],reverse=True)
            ct=[]
            for ctop in range(ntop):
                if len(s)<=ctop: break
                v,i,j=s[ctop]
                ct.append((self.sendocs[i][j],v))
            perTop[t]=ct
        return perTop



    def saveOutputs(self, folder="Bert-SenClu", topWordPerTopic=10, topSenPerTopic=10, topDocsPerTopic=10, maxSenPerDoc=50, addTopicMarkup=True):
        """
        Produce csv and visualization files and store in given folder; The csv can also be used for exploration and visualization
        A topic contains a probability, top words (each with probability), top sentences (each with score), top documents for that topic (full docs, only topic sentences, topic sentences with context); each doch has a score and an ID of the original document in the dataset fed into the topic model)
        For top documents, three representations are output:
        - "FullDoc" The full document up to specified maximum number of sentences maxSenPerDoc
        - "ContextDoc"  The document always contains the first 5 and the last 2 sentences and otherwise only sentences belonging to the topic (plus 1 sentence before/after to give context)
        - "TopicOnlyDoc" Keep only sentences belonging to the document
        The csv file contains 1 row for each topic and each column corresponds to a single item, e.g., a topic word, a topic probability, a top sentence etc.
        -------------------------------------------------
        Input
        fileName ... name of csv file
        topSenPerTopic=15... number of top sentences per topic
        topDocsPerTopic=20 ... number of top documents per topic
        maxSenPerDoc=100 ... maximum number of displayed sentences per document
        addTopicMarkup=True ... for a top document of a topic, highlight sentences belonging to topic
        Returns:
        None (stores files in given folder)
        """
        #get data
        senPerTop=self.getTopSentencesPerTopic(topSenPerTopic)
        docPerTop=self.getTopDocsPerTopic(topDocsPerTopic)

        #Compute  CSV
        header = ["Topic_ID", "Topic_Prob"]
        r=lambda x: str(np.round(x,4))
        if topWordPerTopic>self.maxTopWords:
            print("Exceeded max words per Topic. Fixing to allowed maximum of ",self.maxTopWords)
            topWordPerTopic=self.maxTopWords
        headW=sum([["Word_"+str(i),"Prob_"+str(i)] for i in range(topWordPerTopic)],[])
        headS = sum([["Sentence_" + str(i), "Prob_" + str(i)] for i in range(topSenPerTopic)], [])
        headD = sum([["FullDoc_" + str(i),"ContextDoc_"+ str(i),"TopicOnlyDoc_"+str(i),"Prob_" + str(i),"DocID_"+str(i)] for i in range(topDocsPerTopic)], [])
        rows = []

        def getFixedLenLi(li,nEle):
            conLi = [[w[0], r(w[1])] for w in li[:nEle]]
            conLi=sum(conLi,[])
            return conLi
        topicColor='#D84141'
        for i in range(self.ntopics):
            topdat = [str(i),r(self.pt[i])]
            topW=getFixedLenLi(self.topicsAsWordScoreList[i], topWordPerTopic)
            topS=getFixedLenLi([(("<span style='color:"+topicColor+"'>"  + s[0]  + "</span>") if addTopicMarkup else s[0],s[1]) for s in senPerTop[i]], topSenPerTopic)

            topD=docPerTop[i]
            conDoc=[]
            #Add Docs - keep only sentences of the topic i and for each topic sentence one before and after as well as first and last sentences

            for d,dsc,did in topD:
                topSen=(np.array(self.assignSen_ptd[did]) == i).astype(np.int)
                senToKeep = np.copy(topSen)
                senToKeep[:5]=1 #keep first and last sentences
                senToKeep[-2:] = 1
                senToKeep[:-1]+=topSen[1:] #for each topic sentence add one before and after of a different topic
                senToKeep[1:] += topSen[:-1]
                consens,topsens,alls = [],[],[]
                thres = np.median(self.prob_t[did][topSen>0]) # highlight top 50% of a sentence more
                skipSen=0
                for inds,s in enumerate(d):
                    if topSen[inds]: #Topic sentence?
                        if addTopicMarkup:  # add markup if topic sentence
                            bold = self.prob_t[did][inds] > thres
                            s = "<span style='color:"+topicColor+"'>" + ("<b>" if bold else "") + s + ("</b>" if bold else "") + "</span>" #redish color
                        if len(topsens) < maxSenPerDoc: topsens.append(s)
                    elif addTopicMarkup:  # add markup if topic sentence
                        s = "<span style='color:#7D88C7'>"+ s + "</span>"  # blue color
                    if len(alls) < maxSenPerDoc: alls.append(s)
                    if senToKeep[inds]:
                        if skipSen:  # if left out a sentence add dots
                            s = " ...>>" + str(skipSen) + "... " + s
                        if len(consens) < maxSenPerDoc: consens.append(s)
                        skipSen=0
                    else:
                        skipSen+=1
                conDoc.append([" ".join(alls), " ".join(consens), " ".join(topsens if len(topsens) else [" "]),r(dsc),str(did)])
            conDoc = sum(conDoc, [])
            topdat+=topW+topS+conDoc
            rows.append(topdat)

        df=rows
        df = pd.DataFrame(df, columns = header+headW+headS+headD)
        colW = [c for c in df.columns if c.startswith("Word_")]
        df["TopicShort"] = df.apply(lambda r: "_".join([str(r["Topic_ID"])] + list(r[colW].values[:5]))[:60], axis=1)
        os.makedirs(folder, exist_ok=True)
        df.to_csv(folder+"/topic_all_info.csv")
        nvec=np.transpose(self.vec_t.cpu().numpy())


        #Create visualization of topics
        visUtils.hierarchy(folder, nvec, df["TopicShort"].values)
        visUtils.tsne(self, nvec, folder, df["TopicShort"].values)




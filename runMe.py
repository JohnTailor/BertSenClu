import os

if __name__ == '__main__':
    print(os.getcwd())
    from sklearn.datasets import fetch_20newsgroups
    # import sampleData
    import numpy as np
    from bertSenClu import senClu

    docs = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes')).data #get raw data
    # docs,_= sampleData.get20NewsData() #get data with some pre-processing

    folder = "bertSenCluOutputs/"
    topic_model= senClu.SenClu()
    topics, probs = topic_model.fit_transform(docs, nTopics=20, loadAndStoreInFolder=folder)

    topic_model.saveOutputs(folder)  # Save outputs in folder, i.e. csv-file and visualizations

    #Print Topics
    for it,t in enumerate(topics):
        print("Topic",it,str(t[:10]).replace("'",""))

    print("Top 10 words with scores for topic 0",topic_model.getTopicsWitchScores()[0][:10])
    print("Distribution of topics for document 0", np.round(topic_model.getTopicDocumentDistribution()[0],3))
    print("Distribution of topics", np.round(topic_model.getTopicDistribution(), 3))
    print("First 4 sentences for top doc for topic 0 ", topic_model.getTopDocsPerTopic()[0][0][:4])
    print("Top 3 sentences for topic 0 ", topic_model.getTopSentencesPerTopic()[0][:3])

    import subprocess
    print("Optional: Launching visualization in browser from stored data (can also be called from Shell)")
    subprocess.run("streamlit run visual.py -- --folder "+folder,shell=True)
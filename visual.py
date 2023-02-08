import streamlit as st
import importlib
import argparse,sys
import pandas as pd,os
from streamlit.components.v1 import html

try: # https://github.com/RobertoFN/streamlit-scrollable-textbox #https://discuss.streamlit.io/t/scrolling-text-containers/26485/5
    stx = importlib.import_module("streamlit-scrollable-textbox")
except ModuleNotFoundError:
    import streamlit_scrollable_textbox as stx
    pass


def parse_args(args):
    parser = argparse.ArgumentParser('Bert-SenClu Visualizer')
    parser.add_argument('-f', '--folder', help='folder with Topic Model Output', required=True)
    return parser.parse_args(args)

@st.cache
def get_data(folder):
    def isfloat(num):
        try:
            float(num)
            return True
        except ValueError:
            return False
    csvfile=folder+"/topic_all_info.csv"
    if not os.path.exists(csvfile):
        print("ERROR cannot find the csvfile",csvfile)
        print("Either file is not existing or path to file is incorrect; Default path is ",os.getcwd())
    df=pd.read_csv(csvfile)
    colW=[c for c in df.columns if c.startswith("Word_")]
    df["TopW"] = df.apply(lambda r: ", ".join([str(w) for w in r[colW] if not isfloat(w)])[:2000], axis=1) #filter nans
    return df


def do_stuff_on_page_load(): #https://discuss.streamlit.io/t/set-widemode-by-default/373/2
    st.set_page_config(layout="wide")
do_stuff_on_page_load()
cargs = parse_args(sys.argv[1:])
df = get_data(cargs.folder)


st.title('Bert-SenClu Topic Model')
#st.write("check out this [link](https://share.streamlit.io/mesmith027/streamlit_webapps/main/MC_pi/streamlit_app.py)")
col1, col2, col3 = st.columns([5,4,15])
with col1:
    topicSel = st.radio("Topic", df["TopicShort"])
iTop = int(topicSel.split("_")[0])
with col2:
    st.caption("Top Words")
    st.write(str(df["TopW"].loc[iTop]))
with col3:
    st.caption("Top docs / sentences")
    opts=["Docs with topic sentences","Full docs","Docs with topic sentences and context ","Sentences per topic"]
    topSel = st.radio("View", opts)
    r= df.loc[iTop]
    if topSel==opts[-1]:
        colS = [c for c in df.columns if c.startswith("Sentence_")]
        ltext = "</br> ".join(r[colS])
        print(ltext)
    else:
        collFD = [c for c in df.columns if c.startswith("FullDoc_")]
        collCD = [c for c in df.columns if c.startswith("ContextDoc_")]
        collTD = [c for c in df.columns if c.startswith("TopicOnlyDoc_")]
        ind=collTD if topSel==opts[0] else (collFD if topSel==opts[1] else collCD)
        docs=r[ind]
        docstr="<span style='color:#C0C0C0'> Doc </span></br>"
        ltext=docstr+("</br>"+docstr).join(docs)
    #ltext=ltext.replace("blue","#D84141")
    html(ltext, height=500, scrolling=True)


from PIL import Image
fname=cargs.folder+'/topic_visual_hierarchy.png'
if os.path.exists(fname):
    image = Image.open(fname)
    st.image(image, caption='Topic Hierarchy')
else:
    st.write("Visualization not found. Did you use 'createVisual=True' in saveOutputs?", fname)

fname=cargs.folder+'/topic_visual_tsne.png.png'
if os.path.exists(fname):
    image = Image.open(fname)
    st.image(image, caption='TSNE Clustering')
else:
    st.write("Visualization not found. Did you use 'createVisual=True' in saveOutputs?", fname)

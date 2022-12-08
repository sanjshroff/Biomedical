# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 19:46:19 2022

@author: sanjs
"""
#required modules are imported
from Bio import Medline
import pandas as pd
from tqdm import tqdm
import os
import time
import re
import nltk
import torch
from wordcloud import WordCloud
import matplotlib.pyplot as plt
nltk.download('stopwords')
from nltk.corpus import stopwords
sno = nltk.stem.SnowballStemmer('english') 
stop=set(stopwords.words('english'))
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering 
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import silhouette_samples, silhouette_score
from sentence_transformers import SentenceTransformer,util
#count_vect = CountVectorizer()
model = SentenceTransformer('all-MiniLM-L6-v2')
finalDate = []
input_pubmed = os.getcwd() + r'\data\Pubmed'
input_TREC = os.getcwd() + r'\data\2005trec.csv'

'''
Definition: Function to find the Cosine similarity between sentences within cluster and the user query
input: dataframe containing sentences and cluster number
output: top sentences ordered based on similarity scores   
hard coded to fetch sbert values from cluster 7 only
'''
def sentence_bert(df):
    list_sbert_values = []
    df_sbert = df[df['cluster'] == 7]
    df_sbert=df_sbert[["CleanedText","Title"]]
    df_sbert = df_sbert.reset_index()
    train = df_sbert.iloc[0,1]
    embedding = model.encode(train,convert_to_tensor=True)
    for i in df_sbert.index:
        sentance = df_sbert.iloc[i,1]
        embedding2 = model.encode(sentance,convert_to_tensor=True)
        sim = util.pytorch_cos_sim(embedding,embedding2)
        list_sbert_values.append(sim[0][0])
    df_sbert['Similarity score'] = list_sbert_values
    df_sbert = df_sbert.sort_values(by=["Similarity score"], ascending=False)
    df_sbert.to_csv(os.getcwd()+r'/output/sbert_topk.csv', index= False)

'''
Definition: Function to parse medline to fetch title, abstract, MESH terms
input: file path of text files containing necessary data about each topic
output: a CSV with PMID, Title, Abstract and MESH attached
'''
def fetchInputData (filepath,name):
    finalDate = []
    with open(filepath,encoding = 'utf-8') as f:
        #values_each_pmid = []
        list_of_pmid = Medline.parse(f)
        for pmid in list_of_pmid:
            try:
                pid = pmid['PMID']
            except:
                pip = ''
            try:
                title = pmid['TI']
            except:
                title = ''
            try:
                abstract = pmid['AB']
            except:
                abstract = ''
            try:
                mesh = " ".join(pmid['MH'])
                #print(mesh)
            except:
                mesh = ''
            dict_1 = {'PMID':pid,
               'Title':title,
               'Abstract':abstract,
             'MH':mesh}
            finalDate.append(dict_1)
        df = pd.DataFrame(finalDate)
    return df

'''
Definition: Function to combine title, abstract, MESH terms
input: file path of text files containing necessary data about each topic
output: a CSV with PMID, Title, Abstract and MESH attached
'''
def create_corpus(input_folder):
    list_df =[]
    for input_file in os.listdir(input_folder) :
        #print(input_file)
        cluster_name = input_file.split("\\")[0].split("-")[1]
        x = fetchInputData(input_folder+"\\"+input_file,cluster_name)
        list_df.append(x)
        #print("Got ",len(list_df), " from file",cluster_name)
    final_input_df = pd.concat(list_df)
    final_input_df['Test'] = final_input_df["Title"] + " " + final_input_df["Abstract"] + " " + final_input_df["MH"]
    final_input_df.to_csv(os.getcwd()+r'/data/2022MedlineCombined_test.csv',index = False)
    final_input_df['MH'] = [''.join(map(str, l)) for l in final_input_df['MH']]
    return final_input_df
    
#different format for reading TREC data
def read_trec(input_folder):
    trec_ip = pd.read_csv(input_TREC)
    df= trec_ip
    df['MESH'] = [''.join(map(str, l)) for l in df['MESH']]
    print(df.head())
    return df

#function to clean the word of any html-tags
def cleanhtml(sentence): 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', sentence)
    return cleantext

 #function to clean the word of any punctuation or special characters
def cleanpunc(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    return  cleaned

def apend_clean_text(df):
    str1=' '
    final_string=[]
    for sent in df['Test'].values:
        filtered_sentence=[]
        #print(sent)
        try:
            for w in sent.split():
                for cleaned_words in cleanpunc(w).split():
                    if((cleaned_words.isalpha()) & (len(cleaned_words)>2)):    
                        if(cleaned_words.lower() not in stop):
                            s=(sno.stem(cleaned_words.lower())).encode('utf8')
                            filtered_sentence.append(s)
                        else:
                            continue
                    else:
                        continue 
        except:
            print("failed to load sentence")
        #filtered sentence
        str1 = b" ".join(filtered_sentence) 
        #final string of cleaned words 
        final_string.append(str1)
        
    df['CleanedText']=final_string
    df['CleanedText']=df['CleanedText'].str.decode("utf-8")
    print(df.head())
    return df

def NMI(predicted):
    actual = [i//10 for i in range(len(predicted))]
    print("The NMI score: ", normalized_mutual_info_score(actual, predicted))

def tfidf_kmeans(df):
    cv=CountVectorizer().fit(df['CleanedText']) #change max_features to 200 or None
    sk=cv.transform(df['CleanedText'])
    #tDTM=pd.DataFrame(sk.toarray(),columns=cv.get_feature_names())
    tfmo=TfidfTransformer().fit(sk)
    tfs=tfmo.transform(sk)
    tfDTM=pd.DataFrame(tfs.toarray(),columns=cv.get_feature_names())
    begin_time = time.time()
    # kmeans=KMeans(n_clusters = 10).fit(tfDTM)
    # clus=kmeans.predict(tfDTM)
    model = KMeans(n_clusters = 10,init='k-means++',random_state=99)
    pred_values = model.fit_predict(tfDTM)
    time_taken = time.time() - begin_time
    labels = model.labels_
    pubmed_cl = pd.DataFrame(list(zip(df['Title'], labels)), columns=['title', 'cluster'])
    pubmed_cl.to_csv(os.getcwd()+r'/logs/kmeans_title_with_cluster_num.csv', index= False)
    print("Tfidf kmeans time taken: ", time_taken)
    silhouette_avg = silhouette_score(tfs.toarray(), labels, metric='euclidean')
    print("For n_clusters =", 10,
          "Silhouette score Kmeans:", silhouette_avg)
    NMI(pred_values)
    show_labels(model.labels_)
    #generate_wordcloud(10, pubmed_cl, df['CleanedText'])
    #return model.fit(sk)

def tfidf_aggolomative_clustering(df):
    cv=CountVectorizer().fit(df['CleanedText']) #change max_features to 200 or None
    sk=cv.transform(df['CleanedText'])
    #tDTM=pd.DataFrame(sk.toarray(),columns=cv.get_feature_names())
    tfmo=TfidfTransformer().fit(sk)
    tfs=tfmo.transform(sk)
    tfDTM=pd.DataFrame(tfs.toarray(),columns=cv.get_feature_names())
    begin_time = time.time()
    agg = AgglomerativeClustering(n_clusters=10,affinity='euclidean', linkage='ward')
    pred_values = agg.fit_predict(tfDTM)
    time_taken = time.time() - begin_time
    labels = agg.labels_
    pubmed_cl = pd.DataFrame(list(zip(df['Title'], labels)), columns=['title', 'cluster'])
    df['cluster'] = labels
    df.to_csv(os.getcwd()+r'/output/agglo_with_cluster_num.csv', index= False)
    #print(pubmed_cl.sort_values(by=['cluster']))
    labels = agg.labels_
    print("Tfidf aggolomative time taken: ", time_taken)
    silhouette_avg = silhouette_score(tfs.toarray(), labels, metric='euclidean')
    print("For n_clusters =", 10,
          "The silhouette score Agglomerative:", silhouette_avg)
    NMI(pred_values)
    show_labels(agg.labels_)
    #generate_wordcloud(10, pubmed_cl, df['CleanedText'])
    cluster_labels = agg.fit_predict(tfDTM)
    sentence_bert(df)

def generate_wordcloud(cluster_size, temp_df, cleanText):
    result={'cluster':temp_df['cluster'],'CleanedText': list(cleanText)}
    res=pd.DataFrame(result)
    for k in range(0,cluster_size):
       s=res[res.cluster==k]
       text=s['CleanedText'].str.cat(sep=' ')
       text=text.lower()
       text=' '.join([word for word in text.split()])
       wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
       print('Cluster: {}'.format(k))
       #print('Titles')
       titles=temp_df[temp_df.cluster==k]['title']         
       print(titles.to_string(index=False))
       #plt.figure()
       plt.imshow(wordcloud, interpolation="bilinear")
       plt.axis("off")
       plt.savefig(os.getcwd()+r"/logs/wordcloud/cluster"+str(k)+".png")
       
       # plt.show()
       
def show_labels(str1):
    j=0
    for i in range(9):
        print(str1[j:j+10])
        j+=10
        
#uncomment below lines for TREC
# find ways to display cluster and find evalution matrix for trec
# df_trec = read_trec(input_TREC)
# df_trec = apend_clean_text(df_trec)
# trec_tfidf_aggolomative_clustering(df_trec)
#tfidf_kmeans(df_trec)

#probability scores, disct frm centriud, quantifcatin, relevance, distance between centroid, fartyehrs cluster, scores from 
#kmeans++ , explainpaper.com, plotly 

if __name__ == '__main__':
    #uncomment below line for reading trec data
    df_pubmed = create_corpus(input_pubmed)
    df_pubmed = apend_clean_text(df_pubmed)
    tfidf_aggolomative_clustering(df_pubmed)

    #uncomment to run kmeans clustering
    tfidf_kmeans(df_pubmed)

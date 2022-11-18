# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 19:46:19 2022

@author: sanjs
"""

from Bio import Medline
import pandas as pd
from tqdm import tqdm
import os
import time
import re
import nltk

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
#count_vect = CountVectorizer()


finalDate = []
input_pubmed = os.getcwd() + r'\data\Pubmed'
input_TREC = os.getcwd() + r'\data\2005trec.csv'

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
    #print(len(df))
    return df

def create_corpus(input_folder):
    list_df =[]
    for input_file in os.listdir(input_folder) :
        #print(input_file)
        cluster_name = input_file.split("\\")[0].split("-")[1]
        x = fetchInputData(input_folder+"\\"+input_file,cluster_name)
        list_df.append(x)
        #print("Got ",len(list_df), " from file",cluster_name)
    final_input_df = pd.concat(list_df)
    final_input_df.to_csv(os.getcwd()+r'/data/2022MedlineCombined.csv',index = False)
    df= final_input_df
    df['MH'] = [''.join(map(str, l)) for l in df['MH']]
    return df
    
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
    for sent in df['Abstract'].values:
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
    cv=CountVectorizer().fit(df['CleanedText'])
    sk=cv.transform(df['CleanedText'])
    #tDTM=pd.DataFrame(sk.toarray(),columns=cv.get_feature_names())
    tfmo=TfidfTransformer().fit(sk)
    tfs=tfmo.transform(sk)
    tfDTM=pd.DataFrame(tfs.toarray(),columns=cv.get_feature_names())
    start_time = time.time()
    # kmeans=KMeans(n_clusters = 10).fit(tfDTM)
    # clus=kmeans.predict(tfDTM)
    model = KMeans(n_clusters = 10,init='k-means++', n_jobs = -1,random_state=99)
    pred_values = model.fit_predict(tfDTM)
    time_taken = time.time() - start_time
    labels = model.labels_
    pubmed_cl = pd.DataFrame(list(zip(df['Title'], labels)), columns=['title', 'cluster'])
    pubmed_cl.to_csv(os.getcwd()+r'/logs/kmeans_title_with_cluster_num.csv', index= False)
    print("Tfidf kmeans time taken: ", time_taken)
    silhouette_avg = silhouette_score(tfs.toarray(), labels, metric='euclidean')
    print("For n_clusters =", 10,
          "The average silhouette_score is :", silhouette_avg)
    NMI(pred_values)
    show_labels(model.labels_)
    #generate_wordcloud(10, pubmed_cl, df['CleanedText'])
    #return model.fit(sk)

def tfidf_aggolomative_clustering(df):
    cv=CountVectorizer().fit(df['CleanedText'])
    sk=cv.transform(df['CleanedText'])
    #tDTM=pd.DataFrame(sk.toarray(),columns=cv.get_feature_names())
    tfmo=TfidfTransformer().fit(sk)
    tfs=tfmo.transform(sk)
    tfDTM=pd.DataFrame(tfs.toarray(),columns=cv.get_feature_names())
    start_time = time.time()
    agg = AgglomerativeClustering(n_clusters=10,affinity='euclidean', linkage='ward')
    pred_values = agg.fit_predict(tfDTM)
    time_taken = time.time() - start_time
    labels = agg.labels_
    pubmed_cl = pd.DataFrame(list(zip(df['Title'], labels)), columns=['title', 'cluster'])
    pubmed_cl.to_csv(os.getcwd()+r'/logs/agglo_title_with_cluster_num.csv', index= False)
    #print(pubmed_cl.sort_values(by=['cluster']))
    labels = agg.labels_
    print("Tfidf aggolomative time taken: ", time_taken)
    silhouette_avg = silhouette_score(tfs.toarray(), labels, metric='euclidean')
    print("For n_clusters =", 10,
          "The average silhouette_score is :", silhouette_avg)
    NMI(pred_values)
    show_labels(agg.labels_)
    #generate_wordcloud(10, pubmed_cl, df['CleanedText'])
    cluster_labels = agg.fit_predict(tfDTM)
    
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    #return agg.labels_

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
       plt.figure()
       plt.imshow(wordcloud, interpolation="bilinear")
       plt.axis("off")
       plt.show()

def show_labels(str1):
    j=0
    for i in range(9):
        print(str1[j:j+10])
        j+=10
#uncomment below line for reading trec data
#df = read_trec(input_TREC)

#df_pubmed = create_corpus(input_pubmed)
#df_pubmed = apend_clean_text(df_pubmed)
#tfidf_aggolomative_clustering(df_pubmed)
#tfidf_kmeans(df_pubmed)

def trec_tfidf_aggolomative_clustering(df):
    cv=CountVectorizer(max_features=(100)).fit(df['CleanedText'])
    sk=cv.transform(df['CleanedText'])
    #tDTM=pd.DataFrame(sk.toarray(),columns=cv.get_feature_names())
    tfmo=TfidfTransformer().fit(sk)
    tfs=tfmo.transform(sk)
    tfDTM=pd.DataFrame(tfs.toarray(),columns=cv.get_feature_names())
    for n_clusters in range(45,51):
        # Create a subplot with 1 row and 2 columns
        #fig, (ax1, ax2) = plt.subplots(1, 2)
        #fig.set_size_inches(18, 7)
     
        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        #ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        #ax1.set_ylim([0, len(tfDTM) + (n_clusters) * 10])
    
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
        cluster_labels = clusterer.fit_predict(tfDTM)
        labels = clusterer.labels_
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(tfs.toarray(), labels, metric='euclidean')
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)
    
        # Compute the silhouette scores for each sample
        #sample_silhouette_values = silhouette_samples(tfDTM, cluster_labels)
    
#plt.show()
    
# find ways to display cluster and find evalution matrix for trec
df_trec = read_trec(input_TREC)
df_trec = apend_clean_text(df_trec)
trec_tfidf_aggolomative_clustering(df_trec)
#tfidf_kmeans(df_trec)

#probability scores, disct frm centriud, quantifcatin, relevance, distance between centroid, fartyehrs cluster, scores from 
#kmeans++ , explainpaper.com, plotly 
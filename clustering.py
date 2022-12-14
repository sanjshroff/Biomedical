# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 19:46:19 2022

@author: sanjs
"""
#required modules are importedpi
from Bio import Medline
import pandas as pd
import os
import time
import re
import nltk
import ssl
from sslDisable import ssl_disable
#support MAC users( specifically MAC PRO)
ssl_disable()
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import string
nltk.download('stopwords')
from nltk.corpus import stopwords
sno = nltk.stem.SnowballStemmer('english') 
stop=set(stopwords.words('english'))
list_stopwords = nltk.corpus.stopwords.words("english")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering 
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import silhouette_score
import sklearn.metrics as metrics
from sentence_transformers import SentenceTransformer,util
count_vect = CountVectorizer()
model = SentenceTransformer('all-MiniLM-L6-v2')
finalDate = []
input_pubmed = os.getcwd() + r'/data/Pubmed'
input_TREC= os.getcwd() + r'/data/2005_Trec_genomacis.csv'

'''
Definition: Function to find the Cosine similarity between sentences within cluster and the user query
input: dataframe containing sentences and cluster number
output: top sentences ordered based on similarity scores   
hard coded to fetch sbert values from cluster 7 only
'''
def sentence_bert(df,user_input_pmid):
    
    list_sbert_values = []
    #when no input is given, using cluster number 1 as reference and finding the relevent documents for first document
    cluster_number = 1
    query_set = False

    if user_input_pmid:
        print("User entered PMID:   ", user_input_pmid)
        try:
            cluster_number = df.loc[df['PMID'] == user_input_pmid, 'cluster'].iloc[0]
            print(cluster_number)
            query = df.loc[df['PMID'] == user_input_pmid, 'CleanedText'].iloc[0]
            query_set = True
        except:
            print("Invalid PMID entered Retrieving documents from first available cluster are listed below")
    df_sbert = df[df['cluster'] == cluster_number]
    df_sbert=df_sbert[['CleanedText',"Title"]]
    #if user input is not given take first document in cluster 1 as the query
    if not query_set:
        query = df_sbert.iloc[0,1]
    df_sbert = df_sbert.reset_index()
    embedding = model.encode(query,convert_to_tensor=True)
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
def fetchInputData (filepath):
    finalDate = []
    with open(filepath,encoding = 'utf-8') as f:
        list_of_pmid = Medline.parse(f)
        for pmid in list_of_pmid:
            mesh = ""
            try:
                pid = pmid['PMID']
                title = pmid['TI']
                abstract = pmid['AB']
                mesh = " ".join(pmid['MH'])
            except:
                pass
            dict_values = {'PMID':pid,'Title':title,'Abstract':abstract,'MH':mesh}
            finalDate.append(dict_values)
        df = pd.DataFrame(finalDate)
    return df

'''
Definition: Function to combine title, abstract, MESH terms
input: file path of text files containing necessary data about each topic
output: a CSV with PMID, Title, Abstract and MESH attached
'''
def create_corpus(input_folder, inputList=[]):
    list_df =[]
    for input_file in os.listdir(input_folder) :
        #cluster_name = input_file.split("\\")[0].split("-")[1]
        x = fetchInputData(input_folder+"/"+input_file)
        list_df.append(x)
    final_input_df = pd.concat(list_df)
    if len(inputList) == 4:
        final_input_df.loc[len(final_input_df)] = inputList
    else:
        print("Input criterion not met, either Title/abstract/MESH missing for given PMID")
    final_input_df['Test'] = final_input_df["Title"] + " " + final_input_df["Abstract"] + " " + final_input_df["MH"]
    final_input_df.to_csv(os.getcwd()+r'/data/2022MedlineCombined_test.csv',index = False)
    final_input_df['MH'] = [''.join(map(str, l)) for l in final_input_df['MH']]
    return final_input_df

#function to clean the word of any punctuation
def remove_punctuations(sentence):
    clean_text = sentence.translate(str.maketrans('', '', string.punctuation))
    return  clean_text

'''
Definition: function to remove punctuations, stop words and change all letters to lower case
input: data frame having uncleaned text
output: data frame after cleaning text fields
'''
def apend_clean_text(df):
    str_temp=' '
    final_string=[]
    for sentance in df['Test'].values:
        filtering=[]
        try:
            for word in sentance.split():
                for remove_punct in remove_punctuations(word).split():
                    if((remove_punct.isalpha()) & (len(remove_punct)>2)):    
                        if(remove_punct.lower() not in stop):
                            stemmed=(sno.stem(remove_punct.lower())).encode('utf8')
                            filtering.append(stemmed)
                        else:
                            continue
                    else:
                        continue 
        except:
            print("failed to clean sentence")

        str_temp = b" ".join(filtering) 
        final_string.append(str_temp)
        
    df['CleanedText']=final_string
    df['CleanedText']=df['CleanedText'].str.decode("utf-8")
    return df

# removing null values for the abstract and joining the title,abstract,mesh into test column
def removenull(data):
    data=pd.read_csv(data)
    trec_data=data.dropna().reset_index(drop='True')
    true_labels=trec_data['TOPICID']
    trec_data['Test']=trec_data["Title"] + " " + trec_data["Abstract"] + " " + trec_data["MESH"]
    trec_data_test=trec_data['Test']
    trec_data['MESH']=[''.join(map(str, l)) for l in trec_data['MESH']]
    return trec_data_test,true_labels
    
# Removing null values in the abstract and combing the Title, Abstract and mesh for the Trec dataset
def text_pre_processing(textual_data,stemmer,stopwords=None):
    textual_data.lower()    
    textual_data = re.sub(r'[?|!|\'|"|#]',r'',textual_data)
    textual_data = re.sub(r'[.|,|)|(|\|/]',r' ',textual_data)
    textual_lst = textual_data.split()

    if stopwords is not None:
        textual_lst = [word for word in textual_lst if word not in stopwords ]

    if stemmer == True:
        snowstem = nltk.stem.SnowballStemmer('english')
        textual_lst = [snowstem.stem(word) for word in textual_lst]
    
    textual_data = " ".join(textual_lst)
    return textual_data

# Calculating the nmi score
def NMI(predicted,actual,isTREC):
    if(not isTREC):
        actual = [i//10 for i in range(len(predicted))]
        print("NMI score : ", normalized_mutual_info_score(actual, predicted))
    else:
         print("NMI score : ", normalized_mutual_info_score(actual, predicted))

'''
Definition: function to vectorise text using TF - IDF
input: data frame and mf flag to state maximum features to be used
output: data frame having tmatrix of dense vectors
'''
def tfidf_vectorization(df_col,mf = None):
    count_vectors=CountVectorizer(max_features=mf)
    vectors=count_vectors.fit_transform(df_col)
    transform_vectors=TfidfTransformer().fit(vectors)
    final_vectors=transform_vectors.transform(vectors) 
    return pd.DataFrame(final_vectors.toarray())

'''
Definition: function to cluster based on Kmeans algorithm for PubMed dataset and display cluster performance
input: data frame that contains cleaned data 
'''
def tfidf_kmeans(df):
    tfidf = tfidf_vectorization(df['CleanedText'])
    begin_time = time.time()
    model = KMeans(n_clusters = 10,init='k-means++',random_state=99)
    pred_values = model.fit_predict(tfidf)
    time_taken = time.time() - begin_time
    labels = model.labels_
    pubmed_cl = pd.DataFrame(list(zip(df['Title'], labels)), columns=['title', 'cluster'])
    pubmed_cl.to_csv(os.getcwd()+r'/output/kmeans_title_with_cluster_num.csv', index= False)
    print('******************************************************************************\n')
    print('kmeans on prepared dataset : PubMed Dataset')
    print("Time taken : ", time_taken)
    silhouette_avg = silhouette_score(tfidf, labels)
    print("Silhouette score : ", silhouette_avg)
    NMI(pred_values,None,False)
    show_labels(model.labels_)
    print('******************************************************************************\n')

'''
Definition: function to cluster based on Aggolemarative algorithm for PubMed dataset and display cluster performance
input: data frame that contains cleaned data 
'''
def tfidf_aggolomerative_clustering(df,user_input_pmid = 0):
    tfidf = tfidf_vectorization(df['CleanedText'])
    begin_time = time.time()
    agg = AgglomerativeClustering(n_clusters=10, linkage='ward')
    pred_values = agg.fit_predict(tfidf)
    time_taken = time.time() - begin_time
    labels = agg.labels_
    pubmed_cl = pd.DataFrame(list(zip(df['Title'], labels)), columns=['title', 'cluster'])
    df['cluster'] = labels
    df.to_csv(os.getcwd()+r'/output/agglo_with_cluster_num.csv', index= False)
    labels = agg.labels_
    print('******************************************************************************\n')
    print('Agglomerative on prepared dataset : PubMed Dataset')
    print("Time taken : ", time_taken)
    silhouette_avg = silhouette_score(tfidf, labels)
    print("Silhouette score : ", silhouette_avg)
    NMI(pred_values,None,False)
    show_labels(agg.labels_)
    sentence_bert(df,user_input_pmid)
    print('******************************************************************************\n')
    generate_wordcloud(10,pubmed_cl,df['CleanedText'])

### calculating trec agglomerative clustering for trec data
def trec_agglomerative(cleaned_text,actual):
    print('******************************************************************************\n')
    print('Agglomerative on Ground Truth : TREC Dataset')
    tfidf = tfidf_vectorization(cleaned_text,2000)
    for index, linkage in enumerate(('average', 'complete', 'ward')):
        model = AgglomerativeClustering(linkage=linkage,n_clusters=50,)
        t0 = time.time()
        labels=model.fit_predict(tfidf)
        elapsed_time = time.time() - t0
        print(linkage)
        print("Time taken : ",elapsed_time)
        score= metrics.silhouette_score(tfidf,labels)
        print("Silhouette score : ",score)
        NMI(labels,actual,True)
    print('******************************************************************************\n')
        
### calculating the kmeans clustering for trec data
def trec_kmeans(cleaned_text,actual):
    print('******************************************************************************\n')
    print('kmeans on Ground Truth : TREC Dataset')
    tfidf = tfidf_vectorization(cleaned_text,2000)
    kclustering = KMeans(n_clusters=50,init='k-means++', n_init = 20,random_state=42,)
    t0=time.time()
    labels=kclustering.fit_predict(tfidf)
    elapsed_time = time.time() - t0
    print("Time taken : ",elapsed_time)
    silhouette_score = metrics.silhouette_score(tfidf,labels)
    print("Silhoutte score : ",silhouette_score)
    NMI(labels,actual,True)
    print('******************************************************************************\n')
    
'''
Definition: function to generate word cloud based on clusters and store under logs folder
input: size of the cluster, dataframe containing cluster labels and cleaned text
'''
def generate_wordcloud(cluster_size, temp_df, cleanText):
    result={'cluster':temp_df['cluster'],'CleanedText': list(cleanText)}
    res=pd.DataFrame(result)
    for i in range(0,cluster_size):
       s=res[res.cluster==i]
       text=s['CleanedText'].str.cat(sep=' ')
       text=' '.join([word for word in text.split()])
       wordcloud = WordCloud(max_font_size=40, max_words=75, background_color="white").generate(text)      
       plt.imshow(wordcloud)
       plt.axis("off")
       plt.savefig(os.getcwd()+r"/logs/wordcloud/cluster"+str(i)+".png")
       

'''
Definition: function to display the labels predicted by clustering algorithm
input: string variable containing the predicted values
'''
def show_labels(string_input):
    j=0
    print("Output Labels : ")
    for i in range(9):
        print(string_input[j:j+10])
        j+=10

if __name__ == '__main__':
    #retrieving data from pubmed dataset
    df_pubmed = create_corpus(input_pubmed)
    df_pubmed = apend_clean_text(df_pubmed)
    tfidf_aggolomerative_clustering(df_pubmed)
    tfidf_kmeans(df_pubmed)

    #retrieving data from TREC dataset
    data_trec,actual = removenull(input_TREC)
    cleaned_text=data_trec.apply(lambda x: text_pre_processing(x, True, list_stopwords))
    trec_agglomerative(cleaned_text,actual)
    trec_kmeans(cleaned_text,actual)
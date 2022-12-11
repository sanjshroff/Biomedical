# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 16:53:45 2022

@author: pavans
"""
import os
from tkinter import Tk,Label,Text,Button
from Bio import Entrez
from clustering import create_corpus, tfidf_aggolomerative_clustering, apend_clean_text
user_ip = []

#Save any input from user when a button is clicked
def save_input(ip):
    user_ip.append(ip)

#function to create the main interface
def show_interface():

    #store the PMID entered by user
    def Take_pmid():
            entered_PMID = input_pmid.get("1.0", "end-1c")
            save_input(entered_PMID)
            
    #enter the number of k entered
    def Take_k():
            entered_k = input_k.get("1.0", "end-1c")
            save_input(entered_k)
            #destoy the interface after k value is inserted
            root.destroy()

    #terminate the window when close is clicked        
    def close_window():
        root.destroy()

    root = Tk()
    root.geometry("300x300")
    root.title("MEDLINE")
    
    pmid_label = Label(text = "Enter the PMID here ")

    input_pmid = Text(root, height = 5, width = 25, bg = "White")

    pmid = Button(root, height = 1, width = 20, text ="Enter PMID", command =lambda:Take_pmid())

    input_k= Text(root, height = 2, width = 25, bg = "White")
     
    top_k = Button(root, height = 1, width = 30, text ="Enter number of records to fetch (k)", command = lambda:Take_k())

    quit_button = Button(root, height = 1, width = 20, text ="Exit", command = lambda:close_window())
    
    #placing all elements on the tkinter
    pmid_label.pack( pady=5)
    input_pmid.pack( pady=5)
    pmid.pack( pady=5)
    input_k.pack( pady=5)
    top_k.pack( pady=5)
    quit_button.pack( pady=5)
    root.mainloop()

def fetch_pmid_db(entered_PMID):
    # Fetching PubMed records from online NCBI database
    Entrez.email = 'sanjshroff2@gmail.com'
    print('******************************************************************************\n')
    print('Retrieving PubMed abstract for',entered_PMID)
    allValues = [entered_PMID]
    try: 
        handle = Entrez.efetch(
                    db="pubmed",
                    id=entered_PMID, # Enter 30419345 as trial input 34736317
                    rettype="full",
                    retmode="xml")
        records = Entrez.read(handle)

        for article in records['PubmedArticle']:

            if 'ArticleTitle' in article['MedlineCitation']['Article'].keys():
                title = article['MedlineCitation']['Article']['ArticleTitle']
                print(title)
                allValues.append(title)

            if 'Abstract' in article['MedlineCitation']['Article'].keys():
                abstract = article['MedlineCitation']['Article']['Abstract']
                abstract_text = abstract['AbstractText'][0]
                allValues.append(abstract_text)
            
            if 'MeshHeadingList' in article['MedlineCitation']:
                mesh = article['MedlineCitation']['MeshHeadingList']
                m = ""
                for x in mesh:
                    m = m +" "+ x.get("DescriptorName")
                allValues.append(m)

        return allValues
    except:   
        print("Unable to fetch record for",entered_PMID)
    print('******************************************************************************\n')
if __name__ == '__main__':
    entered_PMID = 0
    user_input_pmid = show_interface()

    #send first input as pmid
    user_input = fetch_pmid_db(user_ip[0])
    input_pubmed = os.getcwd() + r'\data\Pubmed'
    df_pubmed = create_corpus(input_pubmed,user_input)
    df_pubmed = apend_clean_text(df_pubmed)
    tfidf_aggolomerative_clustering(df_pubmed,user_input[0])
    #display_title_scores(int(user_ip[1]))
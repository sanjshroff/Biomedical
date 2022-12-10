# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 16:53:45 2022

@author: sanjs
"""

from tkinter import Tk,Label,Text,Button
from Bio import Entrez
from tkinter_output import display_title_scores

a = []

def save_ip(b):
    a.append(b)
    print("saving a",a,b)

def create_UI():
    def Take_pmid():
            entered_PMID = input_pmid.get("1.0", "end-1c")
            save_ip(entered_PMID)
            print( entered_PMID)
            #root.destroy()
            
    def Take_k():
            entered_k = input_k.get("1.0", "end-1c")
            save_ip(entered_k)
            print( entered_k)
            
    def close_window():
        root.destroy()
    root = Tk()
    root.geometry("300x300")
    root.title("MEDLINE")
    
    l = Label(text = "Enter the PMID here ")
    l2 = Label(text = "Enter the Title here ")
    input_pmid = Text(root, height = 5,
                    width = 25,
                    bg = "White")
     
    input_title = Text(root, height = 5,
                  width = 25,
                  bg = "light cyan")
     
    pmid = Button(root, height = 1,
                     width = 20,
                     text ="Enter PMID",
                     command =lambda:Take_pmid())

    input_k= Text(root, height = 2,
                  width = 25,
                  bg = "White")
     
    top_k = Button(root, height = 1,
                     width = 30,
                     text ="Enter number of records to fetch (k)",
                     command = lambda:Take_k())
    close = Button(root, height = 1,
                     width = 20,
                     text ="Exit",
                     command = lambda:close_window())
     
    l.pack( pady=5)
    input_pmid.pack( pady=5)
    pmid.pack( pady=5)
    input_k.pack( pady=5)
    top_k.pack( pady=5)
    close.pack( pady=5)
    root.mainloop()

def fetch_pmid_db(entered_PMID):
    Entrez.email = 'sanjshroff2@gmail.com'
    
    # Fetching PubMed records from the NCBI Entrez DB.
    print('Retrieving PubMed abstract for',entered_PMID)
    try:
        handle = Entrez.efetch(
                    db="pubmed",
                    id=entered_PMID, #30419345,
                    rettype="full",
                    retmode="xml")
        records = Entrez.read(handle)
        for article in records['PubmedArticle']:
            if 'Abstract' in article['MedlineCitation']['Article'].keys():
                abstract = article['MedlineCitation']['Article']['Abstract']
                abstract_text = abstract['AbstractText'][0]
                print(abstract_text)
        return abstract_text
    except:
        
        print("Unable to fetch record for",entered_PMID)

if __name__ == '__main__':
    entered_PMID = 0
    user_input_pmid = create_UI()
    #print(a)
    new_text = fetch_pmid_db(a[1])
    display_title_scores()


# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 16:53:45 2022

@author: sanjs
"""
import csv
import os
import tkinter as tk

def display_title_scores(input_limit=4):
    root = tk.Tk()
    root.geometry("900x300")

    root.title("Most Relevant Documents")

    with open(os.getcwd()+r"\output\sbert_topk.csv", "r", newline="") as row:
        eachrow = csv.reader(row)
        row_list = list(eachrow)

    #taking minimum of user input and the cluster size to display those many relevent articles 
    input_limit = min(input_limit,len(row_list))

    for i, row in enumerate(row_list):
        if input_limit >= 0:
            for col in [2, 3]:
                tk.Label(root, text=row[col]).grid(row=i, column=col)
        input_limit = input_limit - 1
    root.mainloop()

if __name__ == '__main__':
    user_input_pmid = display_title_scores()
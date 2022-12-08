# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 16:53:45 2022

@author: sanjs
"""
import csv
import os
import tkinter as tk

def display_title_scores():
    root = tk.Tk()
    root.geometry("600x275")
    root.title("Most Relevant Documents")

    with open(os.getcwd()+r"\output\sbert_topk.csv", "r", newline="") as row:
        eachrow = csv.reader(row)
        row_list = list(eachrow)

    for i, row in enumerate(row_list):
        for col in [2, 3]:
            tk.Label(root, text=row[col]).grid(row=i, column=col)
    root.mainloop()

if __name__ == '__main__':
    user_input_pmid = display_title_scores()
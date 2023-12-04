# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 16:51:16 2023

@author: romai
"""
import random
import numpy as np
import math as mp
import bisect
import pickle
import Alias_methode
import re
file_segmented_text1= "C:/Users/romai/Desktop/TP1_TLN/tlnl_tp1_data/alexandre_dumas/La_Reine_Margot.tok"
file_segmented_text2= "C:/Users/romai/Desktop/TP1_TLN/tlnl_tp1_data/alexandre_dumas/Le_comte_de_Monte_Cristo.tok"


def phrased_text(file):

    Cleaned_text=""
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()
    c=text.split("</s> </s> ")
    for i in range(0,len(c)):
        c[i]=c[i].replace("<s>","")
        c[i]=c[i].replace("\n","")
        
    return c


    

def clean_text(file):

    Cleaned_text=""
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()
    for i in text.split():
        if i!="<s>" and i!="</s>":
            Cleaned_text=Cleaned_text+" " +i
    return Cleaned_text

def Build_index(text):
    Index = {}
    index = 1
    Rev_Index={}
    for i in text.split():
        if i not in Index:
            Index[i] = index
            index = index+1 
    for w,i in Index.items():
        Rev_Index[i]=w
    return Index,Rev_Index



def Build_Occ(text,Index): 
    Occ={}
    for i in text.split():
        if Index[i] in Occ:
            Occ[Index[i]]+=1
        else:
            Occ[Index[i]]=1
    return Occ

def subsampled_text(text,t,Index,Occ):
    sub_text=text.copy()
    for j in range(0,len(text)):
        f_sub = text[j]
        f=""
        for i in f_sub.split():
            p_wi = 1 - np.sqrt(t / Occ[Index[i]])
            r = random.uniform(0, 1)
            if r <p_wi:
                f=f+" " +i
        sub_text[j]=f
    return sub_text


def buil_Occ_frec(min_count,alpha,Occ):
    Occ_freq={}
    sum_c=0
    s=0
    for w,c in Occ.items():
        if c>=min_count:
            Occ_freq[w]=c**alpha
            sum_c+=c**alpha

    for w,c in Occ_freq.items():
        Occ_freq[w]=c/sum_c
    for w,c in Occ_freq.items():
        s+=c
    
    return Occ_freq



def set_proba(Occ_freq):
    L=[0.0]*14000
    for i,p in Occ_freq.items():
        L[i]=p
    return L

def build_data(t,alpha):
    cleaned_text=clean_text(file_segmented_text2)
    ind=Build_index(cleaned_text)
    Index = ind[0]
    Rev_Index=ind[1]
    
    Occ= Build_Occ(cleaned_text,Index)
    Occ_freq=buil_Occ_frec(10,alpha,Occ)
    
    text_phrases= phrased_text(file_segmented_text2)
    sub_text=subsampled_text(text_phrases,t,Index,Occ)

   
    proba=set_proba(Occ_freq)
    
    Alias_set=Alias_methode.alias_setup(proba)
    J=Alias_set[0]
    q=Alias_set[1]
    # Sauvegarde de la liste dans un fichier
    with open('Index.pkl', 'wb') as f:
        pickle.dump(Index, f)
    with open('sub.pkl', 'wb') as f:
        pickle.dump(sub_text, f)
    with open('text_phrase.pkl', 'wb') as f:
        pickle.dump(text_phrases, f)
    with open('Occ.pkl', 'wb') as f:
        pickle.dump(Occ, f)
    with open('Occ_freq.pkl', 'wb') as f:
        pickle.dump(Occ_freq, f)
    with open('Alias_list.pkl', 'wb') as f:
        pickle.dump(J, f)
    with open('Proba_list.pkl', 'wb') as f:
        pickle.dump(q, f)

def get_Index():
    with open('Index.pkl', 'rb') as f:
        I= pickle.load(f)
    return I
def get_sub():
    with open('sub.pkl', 'rb') as f:
        s= pickle.load(f)
    return s
def get_phrased_text():
    with open('text_phrase.pkl', 'rb') as f:
        tf= pickle.load(f)
    return tf
def get_Occ_dic():
    with open('Occ.pkl', 'rb') as f:
        Occ= pickle.load(f)
    return Occ
def get_Occ_freq():
    with open('Occ_freq.pkl', 'rb') as f:
        Occ_freq= pickle.load(f)
    return Occ_freq
def get_Alias_list():
    with open('Alias_list.pkl', 'rb') as f:
        Alias_list= pickle.load(f)
    return Alias_list
def get_Proba_list():
    with open('Proba_list.pkl', 'rb') as f:
        Proba_list= pickle.load(f)
    return Proba_list




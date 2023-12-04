# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 16:51:16 2023

@author: romai
"""
import random
import numpy as np
import math as mp
import bisect
import prep_data_tp2TL
import Alias_methode
import time
import matplotlib.pyplot as plt
from scipy import stats


file_segmented_text1= "C:/Users/romai/Desktop/TP1_TLN/tlnl_tp1_data/alexandre_dumas/La_Reine_Margot.tok"
file_segmented_text2= "C:/Users/romai/Desktop/TP1_TLN/tlnl_tp1_data/alexandre_dumas/Le_comte_de_Monte_Cristo.tok"


def K_fold(text_phrase):
    Fold=[]
    random.shuffle(text_phrase)
    for i in text_phrase:
        r=random.randint(1, 10)
        if r>1:
            Fold.append(i)
    return Fold

def count(text_phrase):
    count=0
    for i in text_phrase:
        for j in i.split():
            count+=1
            
    return count
            

    


def sig(m, c):
    z = np.dot(m, c)
    if z > 20:  # pour de très grandes valeurs de z, la sigmoïde est ~1
        return 1.0 - 1e-7
    elif z < -20:  # pour de très petites valeurs de z, la sigmoïde est ~0
        return  1e-7
    else:
        return 1.0 / (1.0 + np.exp(-z))
   
def Distrib(Occ_freq):

    l=0
    while True:
        r=random.uniform(0,1)
        s=0
        for i,w in Occ_freq.items():
            s=s+w
            if s>r: 
            
                
                return i
        
 

def add_exemple_p_and_n(dic, FG, L, text, Index,k,min_count,Occ,Occ_freq,J,q):
    
    word = FG[L]
    if word!=0:
        if Occ[word]>=min_count:
            for i in FG:
                ex = []
                if i!=0 and i != word and Occ[i]>=min_count :
                    ex.append(i)
                    for j in range(0,k):
                        while True:
                            #word_neg = Distrib(Occ_freq)
                            word_neg = Alias_methode.alias_draw(J, q)
                            if word_neg != word:
                                ex.append(word_neg)
                                break
                if ex!=[]:
                    if word in dic:
                        dic[word].append(ex)
                    else:
                        dic[word] = [ex]

     
        
def exemple_apprentissage(text, L,Index,K,min_count,Occ,Occ_freq,J,q):
    count=0
    Exemples = {}
    
   
    for ph in text:
    
        Fenetre_glissante = [0]*(2*L+1)
        k = 0
        for i in ph.split():
                    Fenetre_glissante[L+k] = Index[i]
                    k = k+1
                    if k == L+1:
                        break
      
        add_exemple_p_and_n(Exemples, Fenetre_glissante, L, text, Index,K,min_count,Occ,Occ_freq,J,q)
        l = 0
        for i in ph.split():
                count+=1
                l = l+1
                
                if l >= 2*L:
                    for j in range(0, 2*L):
                        Fenetre_glissante[j] = Fenetre_glissante[j+1]
                    Fenetre_glissante[2*L] = Index[i]
                    add_exemple_p_and_n(Exemples, Fenetre_glissante, L, text, Index,K,min_count,Occ,Occ_freq,J,q)
              
        for j in range(0,L):
                for j in range(0, 2*L):
                    Fenetre_glissante[j] = Fenetre_glissante[j+1]
                Fenetre_glissante[2*L] = 0
                add_exemple_p_and_n(Exemples, Fenetre_glissante, L, text, Index,K,min_count,Occ,Occ_freq,J,q)
    return Exemples

#Exemples=exemple_apprentissage(texte_phrase,2,Index,10,12,Occ,Occ_freq)

def Apprentissage(Index, d, eta, L, K, Occ, min_count, nbr_iter, Exemples,Occ_freq):
    Plongements_m={}
    Plongements_c={}
    Sc=[]
    V =len(Occ_freq)
    # Initialisation des plongements avec une dimension supplémentaire pour le biais
    for i in Occ_freq:
        Plongements_m[i]= np.random.uniform(-0.5 / d, 0.5 / d, (d))
        Plongements_c[i] = np.random.uniform(-0.5 / d, 0.5 / d, (d))
    sc=eval_(file_test,Plongements_m,Index)
    Sc.append(sc)
    
    
 
    
    sc=0
    sc1=0

    for nb in range(nbr_iter):
        print(nb)
        for w, E in Exemples.items():
            
            p = Plongements_m[w]

            for e in E:
                ep = e[0] # exemple positif
                cpos = Plongements_c[ep]
                gradient_pos = (sig(p, cpos) - 1)


                Plongements_m[w] -= eta * gradient_pos * cpos
                Plongements_c[ep] -= eta * gradient_pos * p

                sum_cneg = 0
                for en in e[1:]:
                    cneg = Plongements_c[en]
                    gradient_neg = sig(p, cneg)


                    sum_cneg += gradient_neg * cneg
                    Plongements_c[en] -= eta * gradient_neg * p

                Plongements_m[w] -= eta * sum_cneg
        sc1=sc
        sc=eval_(file_test,Plongements_m,Index)
        print(sc)
        Sc.append(sc)
    return Plongements_m,Sc

#pl= Apprentissage(Index,150,0.005,2,10,Occ,12,5,Exemples)
#print(pl)


 
def cosine_similarity(a, b):
    # Calcul de la similarité cosinus
    dot_product = np.dot(a, b)
    
    similarity = dot_product /(np.linalg.norm(a) * np.linalg.norm(b))
    
    return similarity


    
file_test= "C:/Users/romai/Desktop/Le_comte_de_Monte_Cristo.100.sim"

def find_min_count(file_test,Occ,Index):
    with open(file_test, "r", encoding="utf-8") as f:
        text = f.read()
    t = text.split("\n")
    pos=0
    mc=100000000
    for i in t:
            I=i.split()
            if I!=[]:
                for j in I:
                    if Occ[Index[j]]<mc:
                        mc=Occ[Index[j]]
    return mc
                

def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:  # éviter de diviser par zéro
        return v
    return v / norm

def eval_(file_test, pl,Index):
    L = 0
    
    with open(file_test, "r", encoding="utf-8") as f:
        text = f.read()
        t = text.split("\n")
        pos = 0
        
        for i in t:
            I = i.split()
            if I != []:
                vec1 = pl[Index[I[0]]]
                vec2 = pl[Index[I[1]]]
                vec3 = pl[Index[I[2]]]

                if cosine_similarity(vec1, vec2) > cosine_similarity(vec1, vec3):
                    pos += 1
        return pos / len(t)

def plot(Scores,t,alpha):
    S1=[]
    S2=[]
    S3=[]
    S4=[]
    S5=[]
    S6=[]
    S7=[]
    S8=[]
    S9=[]
    S10=[]
    S11=[]
    S12=[]
    S13=[]
    S14=[]
    S15=[]
    for i in Scores:
        S1.append(i[0])
        S2.append(i[1])
        S3.append(i[2])
        S4.append(i[3])
        S5.append(i[4])
        S6.append(i[5])
        S7.append(i[6])
        S8.append(i[7])
        S9.append(i[8])
        S10.append(i[9])
        S11.append(i[10])
        S12.append(i[11])
        S13.append(i[12])
        S14.append(i[13])
        S15.append(i[14])



    N=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]

    # Calculer la moyenne et l'écart-type
    # Calculer les moyennes et les intervalles de confiance pour chaque série de valeurs
    data=[S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S12,S13,S14,S15]
    means = [np.mean(d) for d in data]
    truncated_means=means.copy()
    for i in range(0,len(means)):
        
        truncated_means[i]=round(means[i],3)
    # Calcul de l'intervalle de confiance à 95%
    cis = [stats.norm.interval(0.95, loc=mean, scale=np.std(d)/np.sqrt(len(d))) for d, mean in zip(data, means)]
    
    # Calcul des erreurs pour la fonction errorbar (distance entre les moyennes et les bornes de l'intervalle de confiance)
    errors = [(mean-ci[0], ci[1]-mean) for mean, ci in zip(means, cis)]
    lower_errors = [mean - ci[0] for mean, ci in zip(means, cis)]
    upper_errors = [ci[1] - mean for mean, ci in zip(means, cis)]
    
    # Tracer la courbe des moyennes avec des moustaches pour les intervalles de confiance

    # Generate the error bar plot
    print(len(N))
    print(len(truncated_means))
    plt.errorbar(N, truncated_means, yerr=[lower_errors, upper_errors], fmt='o', linestyle='-', capsize=5, capthick=2, ecolor='black', label='L=2, eta=0.1, d=100, k=10, min_count=10')



# Adding means values above each point

    for (i, mean) in zip(N, truncated_means):

        plt.text(i, mean, f'{mean:.3f}', ha='center', va='bottom')

    
    plt.xlabel("Nombres d'itérations")
    plt.ylabel('Scores moyens')
    plt.title(f'Scores W2V, whithout subsamplig and alpha={alpha}')
    plt.legend()
    plt.show()

def w2v():
   

    d=100
    eta=0.1
    T=[10**-2]
    A=[1]

    for t in T: 
            for alpha in A:
                    prep_data_tp2TL.build_data(t, alpha)
                    Index=prep_data_tp2TL.get_Index()
                    Occ=prep_data_tp2TL.get_Occ_dic()
                    Occ_freq=prep_data_tp2TL.get_Occ_freq()
                    J=prep_data_tp2TL.get_Alias_list()
                    q=prep_data_tp2TL.get_Proba_list()
                    texte_phrase=prep_data_tp2TL.get_phrased_text()
                    sub_text=prep_data_tp2TL.get_sub()
                    Exemples=exemple_apprentissage(texte_phrase,2,Index,10,10,Occ,Occ_freq,J,q)
                    Scores=[]
                    
                    
                    for i in range(0,5):
                        pl= Apprentissage(Index,d,eta,2,10,Occ,10,15,Exemples,Occ_freq)
                        Emb=pl[0]
                        SC=pl[1]
                        
                        Scores.append(SC)
                    plot(Scores,t,alpha)
    return Scores

w2v()



            


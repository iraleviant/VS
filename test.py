import ConfigParser
import numpy
import codecs
import sys
import time
import random
import math
import scipy
import os
from copy import deepcopy
import json
from numpy.linalg import norm
from numpy import dot
from scipy.stats import spearmanr
import scipy.sparse as ss
from sklearn.metrics.pairwise import pairwise_distances
import sklearn.preprocessing as pp
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import pairwise_kernels
import numpy as np
from sklearn.preprocessing import  normalize
from sklearn.metrics.pairwise import cosine_similarity
from scipy  import sparse

def main():
    
    A =  np.array([[0, 1, 0, 0, 1], [0, 0, 1, 1, 1],[1, 1, 0, 1, 0]])
    A_sparse = sparse.csr_matrix(A)

    similarities = cosine_similarity(A_sparse)
    print('pairwise dense output:\n {}\n'.format(similarities))

    #also can output sparse matrices
    similarities_sparse = cosine_similarity(A_sparse,dense_output=False)
    print('pairwise sparse output:\n {}\n'.format(similarities_sparse))
    
    #######################################################################################################################
    mat=ss.load_npz('/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/smooth_ppmi_all_pats_mat_200.npz')
    similarities_mat = cosine_similarity(mat)
    
    
    dic_file_order='/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/cws_dictionary_allpats_python_order_200.dat'
    fread=codecs.open(dic_file_order)
    cws_clean={}
    
    lines_f = fread.readlines()[1:]
    for line_g in lines_f:
        line_f=line_g.strip()
        line=line_f.split(" ")
        cws_clean[line[0]]=line[1]
    print "Finished reading content word dictionary its length is:", len(cws_clean)
        
    #mat=ss.load_npz('/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/proj_mat_250k_gaus.npz') #vectors mat
    
    
    fread_human=codecs.open("/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/english_simlex_word_pairs_human_scores.dat", 'r', 'utf-8')  #scores from felix hill
    
    pair_list_human=[]
    lines_f = fread_human.readlines()[1:]  # skips the first line of "number of vecs, dimension of each vec"
    for line_f in lines_f:
        tokens = line_f.split(",")
        word_i = tokens[0].lower()
        word_j = tokens[1].lower()
        score = float(tokens[2])  
        
        if word_i in cws_clean and word_j in cws_clean:
            pair_list_human.append(((word_i, word_j), score))
        else:
            pass
        
    pair_list_human.sort(key=lambda x: - x[1])  ###sorts the list according to the human scores in descreasing order
    coverage = len(pair_list_human)

    model_list = [] #list:[index, ((word1,word2), model_score) ]
    model_scores = {} #{key=(word1,word2), value=cosine_sim_betwen_vectors}
    
    
    
    
    print "i'm here"
#################################################################     
########################

if __name__ == '__main__':
    main()

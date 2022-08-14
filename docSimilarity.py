import os
import re
import pandas as pd
import tensorflow as tf
import numpy as np
import PyPDF2
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import LSTM
from keras.callbacks import LambdaCallback
from collections import Counter
tok = Tokenizer()
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import nltk
import warnings
warnings.filterwarnings("ignore")

#remove the punctuation
def remove(text):
    remove_chars = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    return re.sub(remove_chars, '', text)

def text_extract(path):
    pdfFileObj = open(path, 'rb')
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
    print("Number of Pages in the Document: ", pdfReader.numPages) 
    pageObj = pdfReader.getPage(0) 
    f = pageObj.extractText()

    print("Extracted Text --->")
    print(f)
    print("\n")
    pdfFileObj.close()
    f=f.lower()
    f=f.replace('\n',' ').replace('\r','')
    f=f.split()
    return f

def read_text_byphase(f):
    data=[]
    f = f.split('\n')
    for lines in f:
        sentence=[]
        lines=lines.lower()
        lines=lines.strip("\n")
        #result_list = re.split(r'[.]', lines)
        result_list = lines.split('.')
        for se in (result_list):
            #print(se)
            se=remove(se)
            if se and se != ' ':
                se=se.split()
                sentence.append(se)
        data.append(sentence)
    return data

def findit(data,keywords):
    res=[]
    for i,phase in enumerate(data):
        for j,line in enumerate(phase):
            for k,word in enumerate(line):
                if word == keywords:
                    res.append([i,j,k])
    return res

def get_difference_result(path_first,path_second,keywords):
    file1 = open(path_first, 'rb') 
    file2 = open(path_second, 'rb')
    f = PyPDF2.PdfFileReader(file1) 
    f2 = PyPDF2.PdfFileReader(file2)  
    f = f.getPage(0) 
    f2 = f2.getPage(0) 
    f = f.extractText()
    f2 = f2.extractText()
    file1.close()
    file2.close()
    data = read_text_byphase(f)
    data2= read_text_byphase(f2)
    res1= findit(data, keywords)
    res2= findit(data2, keywords)
    common_word=[]
    for m in range(len(res1)):
        word=res1[m]
        if word in res2:
            common_word.append(word)
    for n in range(len(common_word)):
        c_word=common_word[n]
        res1.remove(c_word)
        res2.remove(c_word)
    
    if res1:
        for item in res1:
            ph=item[0]+1
            line=item[1]+1
            index=item[2]+1
            print('The document deletes the old word "{}" at Phase {} , Line {} & Index {}.'.format(keywords,ph,line,index))
            # return [keywords,ph,line,index]
            # return a
    
    if res2:
        for item in res2:
            ph=item[0]+1
            line=item[1]+1
            index=item[2]+1
            print('The document adds the new word "{}" at Phase {} , Line {} & Index {}.'.format(keywords,ph,line,index))
            # return [keywords,ph,line,index]

def process(file):
    words=[words.lower() for words in file]
    porter= nltk.PorterStemmer()
    stemmed_tokens=[porter.stem(t) for t in words]
    #count words
    count=nltk.defaultdict(int)
    for word in words:
        count[word]+=1
    # print(count)
    return count;

def cos_sim(a,b):
    dot_product=np.dot(a,b)
    norm_a=np.linalg.norm(a)
    norm_b=np.linalg.norm(b)
    return dot_product/(norm_a * norm_b)

def getSimilarity(dict1,dict2):
    all_words_list=[]
    for key in dict1:
        all_words_list.append(key)
    for key in dict2:
        all_words_list.append(key)
    all_words_list_size=len(all_words_list)

    v1=np.zeros(all_words_list_size,dtype=np.int)
    v2=np.zeros(all_words_list_size,dtype=np.int)
    i=0
    for (key) in all_words_list:
        v1[i]=dict1.get(key,0)
        v2[i]=dict2.get(key,0)
        i=i+1
    return cos_sim(v1,v2)

path1='genuine.pdf'
path2='forged.pdf'

text1 = text_extract(path1)
text2 = text_extract(path2)

dict1=process(text1)
dict2=process(text2)
print("Similarity between two text documents", getSimilarity(dict1,dict2))
print('\n')

counter1=Counter(text1)
dictionary1=dict(counter1)
counter2=Counter(text2)
dictionary2=dict(counter2)
print("Bag Of Words ---> Document 1")
print(dictionary1)
print("\n")
print("Bag Of Words ---> Document 2")
print(dictionary2)
print('\n')

differ = set(dictionary1.items()) ^ set(dictionary2.items())
differ=list(differ)

words=[]
for i in range(len(differ)):
    if differ[i][0] not in words:
        words.append(differ[i][0])

print("Words that were tampered with:\n", words)
print('\n')

print("Text Modifications That Were Detected:")
for w in words:
    get_difference_result(path1, path2, w)
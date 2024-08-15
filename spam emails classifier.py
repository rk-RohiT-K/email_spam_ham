import numpy as np 
import pandas as pd 
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import timeit
"""

"""
"""
Load Data from csv file which contains the labeling data set 
""" 
def load_data(filename):
    df = pd.read_csv(filename)
    df['split'] = np.random.randn(df.shape[0], 1)
    msk = np.random.rand(len(df)) <= 0.7   
    train = df[msk]
    test = df[~msk]
    x_train = train['text']
    y_train = train['spam']
    x_test = test['text']
    y_test = test['spam']
    return x_train, y_train, x_test, y_test


"""
 functionality : given spam emails, build dictionary contrain word and its frequency 
 inputs : x list of emails , y list of labels of emails
 outputs: return the dictionary contain only the words in spam emails and them frequencies
 
"""
def build_dictionary(x,y):
    dic={}
    for email,label in zip(x, y): 
        if label==1:
            for word in email:
                if word not in dic:
                    dic[word] = 1
                else:
                    dic[word]+=1
    return dic

"""
preorocessing data : tokenization , and remove stop words from emails
"""
def get_tokenized(emails):
    stop_words = set(stopwords.words('english'))
    tokenized_list=[]
    for email in emails:
        email_tokenized = [word for word in nltk.word_tokenize(email) if word not in stop_words]
        tokenized_list.append(email_tokenized)
    return tokenized_list

"""
return most common frequency words in dictionary 
inputs: the dictionary of words and their frequency 
"""
def most_common(dic,n):
    dic = Counter(dic)
    dic = dic.most_common(n)
    return dic

"""
Remove stop words and clean the dictionary from unrelated classified words

"""
# remove unrelated word from dictionary 
def clean_dic(dic):  
    list_to_remove = dic.keys()
    for item in list(dic):
        if item.isalpha() == False: 
            del dic[item]
        elif len(dic) <= 1:
            del dic[item]
        elif item=='Subject':
            del dic[item]
    return dic

"""
feature extraction
 functionality : get the features matrix 
 inputs : x refers to emails and attributes refer to most common words in spam emails ,until now 
 outputs : return features_matrix whose rows are emails and colums are common words or features 
"""
def get_features_matrix(x,y,attributes):
    features_matrix = np.zeros((len(x),len(attributes)))
    i = 0
    for email,label in zip(x, y):
        if label==1:
            j = 0
            for word in attributes:
                for word_email in email.split():
                    if word[0]==word_email:
                        features_matrix[i][j]+=1
                j+=1
            i+=1
    return features_matrix

if __name__ == '__main__':
    #calc the time starting from here 
    start = timeit.default_timer()
    #the file contains the dataset 
    filename = 'emails.csv'
    
    #load data 
    print("loading data")
    x_train,y_train,x_test,y_test = load_data(filename)
    #tokenize 
    print("preprocessing data, tokenization..")
    tokenized = get_tokenized(x_train)
    # build dictionary of words 
    print("build dictionary")
    dic = build_dictionary(tokenized, y_train)
    #clean dictionary 
    print("clean dictionary...")
    dic = clean_dic(dic)
    print("most common words")
    dic = most_common(dic,1500)
    
    print("feature extraction...")
    training_features = get_features_matrix(x_train, y_train, dic)
    testing_features = get_features_matrix(x_test, y_test, dic)
    
    
    print("training model ...")
    model = GaussianNB()
    model.fit(training_features, y_train)
   
    print("predction on the test set\n")
    predicted_labels = model.predict(testing_features)
    #end time 
    stop = timeit.default_timer()
    print("ooohhhhooo, accuracy is ", accuracy_score(predicted_labels, y_test)*100,"%")
    print("confusion matrix\n",confusion_matrix(y_test, predicted_labels))
    print("time runing = ",stop-start,"sec")
    
   


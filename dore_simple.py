'''
    dodf_simple

    Active learning with Support Vector Machine
    baseado no trabalho do aluno Hichemm Khalid Medeiros
'''
import re
import pandas as pd
#import matplotlib
#import os
#from time import time
import numpy as np
#import pylab as pl
#from sklearn.feature_selection import SelectKBest, chi2
#from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
#from sklearn import svm
#from sklearn.utils.extmath import density
from sklearn import metrics
#from sklearn.model_selection import cross_validate
#from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
#import itertools
#import shutil
import matplotlib.pyplot as plt


def cleanText(text):
    '''Normalização do texto retirando acentuação, caracteres especiais,
       espaços adicionais e caracteres não textuais'''

    text = text.lower()
    text = re.sub(r"ú", "u", text)
    text = re.sub(r"á", "a", text)
    text = re.sub(r"é", "e", text)
    text = re.sub(r"í", "i", text)
    text = re.sub(r"ó", "o", text)
    text = re.sub(r"u", "u", text)
    text = re.sub(r"â", "a", text)
    text = re.sub(r"ê", "e", text)
    text = re.sub(r"ô", "o", text)
    text = re.sub(r"à", "a", text)
    text = re.sub(r"ã", "a", text)
    text = re.sub(r"õ", "o", text)
    text = re.sub(r"ç", "c", text)
    text = re.sub(r"\\W", " ", text)
    text = re.sub(r"\\s+", " ", text)
    text = text.strip(' ')
    return text


def benchmark():
    '''Processamento do conjunto de treinamento e escolha dos exemplos a serem rotulados'''

    clf.fit(X_train, y_train)

    confidences = clf.decision_function(X_unlabeled)

    df = pd.DataFrame(clf.predict(X_unlabeled))
    df = df.assign(conf = confidences.max(1))
    df.columns = ['int_label', 'conf']
    df.sort_values(by=['conf'], ascending=False, inplace=True)
    question_samples = []

    low_confidence_samples = df.conf.index[0:NUM_QUESTIONS]
    question_samples.extend(low_confidence_samples.tolist())

    # for category in categories:
    #     low_confidence_samples = df[df.int_label == category].conf.index[0:NUM_QUESTIONS]
    #     question_samples.extend(low_confidence_samples.tolist())
    #     df.drop(index=df[df.int_label == category][0:NUM_QUESTIONS].index, inplace=True)

    return question_samples


def clfTest():
    '''Faz as classificacoes e mostra a f1-score resultante'''

    pred = clf.predict(X_test)

    print("classification report:")
    print(metrics.classification_report(y_test, pred))

    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))

    return metrics.f1_score(y_test, pred, average='micro')



clf = SVC()

# Quantidade de requisicoes de rotulos para o oraculo que serao feitas por vez
NUM_QUESTIONS = 100

PATH_TRAIN = "dore_2009_proc.csv"
ENCODING = 'utf-8'
result_x = []
result_y = []

df = pd.read_csv(PATH_TRAIN,encoding = ENCODING,header = 0)
df['conteudo'] = df['conteudo'].map(lambda com: cleanText(com))

categories = df.int_label.unique()

"""Divisao do dataset entre informacoes de treinamento e teste:"""

df_test = df.sample(frac = 0.33, random_state = 1)
# df_test = df.sample(n=230, random_state = 1)

df_train = df.drop(index = df_test.index)

df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

MAX_SIZE = df_train.int_label.size

"""Cria o dataframe com os exemplos rotulados:

*    Seleciona um exemplo para cada rotulo
"""

df_labeled = pd.DataFrame()

for category in categories:
    df_labeled = df_labeled.append( df_train[df_train.int_label==category][0:1], ignore_index=True )
    df_train.drop(index = df_train[df_train.int_label==category][0:1].index, inplace=True)

df_unlabeled = df_train

df_train = df_labeled

# Active learning : loop

while True:

    y_train = df_train.int_label
    y_test = df_test.int_label

    df_unlabeled = df_unlabeled.reset_index(drop=True)

    vectorizer = TfidfVectorizer(encoding= ENCODING, use_idf=True, norm='l2', binary=False, sublinear_tf=True,min_df=0.0001, max_df=1.0, ngram_range=(1, 3), analyzer='word', stop_words=None)

    X_train = vectorizer.fit_transform(df_train.conteudo)
    X_test = vectorizer.transform(df_test.conteudo)
    X_unlabeled = vectorizer.transform(df_unlabeled.conteudo)

    df_unified = df_train.append(df_unlabeled)
    X_unified  = vectorizer.transform(df_unified.conteudo)

    question_samples = benchmark()
    result_x.append(clfTest())
    result_y.append(df_train.int_label.size)

    print('Labeled examples: ', df_train.int_label.size)

    if (df_train.int_label.size < MAX_SIZE - (NUM_QUESTIONS + 1)) and ((len(result_x) < 2) or ( (result_x[-1] - result_x[-2] > -1) or (result_x[-1] < result_x[-2]) )):
        insert = {'int_label':[], 'conteudo':[]}
        cont = 0
        for i in question_samples:
            try:
                insert["int_label"].insert(cont, df_unlabeled.int_label[i])
                insert["conteudo"].insert(cont, df_unlabeled.conteudo[i])
                cont += 1
                df_unlabeled = df_unlabeled.drop(i)
            except Exception as e:
                print("Error:", e)

        df_insert = pd.DataFrame.from_dict(insert)
        df_train = df_train.append(df_insert, ignore_index=True, sort=False)

    else:
        result_y_active = result_y
        result_x_active = result_x
        plt.plot(result_y_active, result_x_active, label='Active learning')
        #plt.plot(result_y_spv, result_x_spv,label = 'Convencional')
        plt.axis([0, 4000, 0.3, 1.0])
        plt.legend(loc='lower right', shadow=True, fontsize='x-large')
        plt.grid(True)
        plt.xlabel('Training set size')
        plt.ylabel('f1-score')
        plt.title('Documents set')
        plt.show()

        result = pd.DataFrame(result_y)
        result = result.assign(y=result_x)
        np.savetxt('results.txt', result, fmt='%f')

        break

    # end

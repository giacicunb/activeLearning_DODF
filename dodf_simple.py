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
from sklearn.calibration import CalibratedClassifierCV
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


def benchmarkBvSB():
    '''Processamento do conjunto de treinamento e escolha dos exemplos a serem rotulados utilizando o método Best vs Second Best'''

    clf.fit(X_train, y_train)

    print('Labeled examples: ', df_train.label.size)
    # Para cada instância, obtém as probabilidades de pertencer a cada classe
    probabilities = clf.predict_proba(X_unlabeled)

    BvSB = []
    for list in probabilities:
        list = list.tolist()
        # Obtém a probabilidade da instância pertencer à classe mais provável
        best = list.pop(list.index(max(list)))
        # Obtém a probabilidade da instância pertencer à segunda classe mais provável
        second_best = list.pop(list.index(max(list)))
        # Calcula a diferença e adiciona à lista
        BvSB.append(best-second_best)

    df = pd.DataFrame(clf.predict(X_unlabeled))
    df = df.assign(conf = BvSB)
    df.columns = ['label', 'conf']
    df.sort_values(by=['conf'], ascending=True, inplace=True)
    question_samples = []

    for category in categories:
        low_confidence_samples = df[df.label == category].conf.index[0:NUM_QUESTIONS]
        question_samples.extend(low_confidence_samples.tolist())
        df.drop(index=df[df.label == category][0:NUM_QUESTIONS].index, inplace=True)

    return question_samples


def clfTest():
    '''Faz as classificacoes e mostra a f1-score resultante'''

    pred = clf.predict(X_test)

    print("classification report:")
    print(metrics.classification_report(y_test, pred))

    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))

    return metrics.f1_score(y_test, pred, average='micro')



clf = LinearSVC(loss='squared_hinge', penalty='l2', dual=False, tol=1e-3, class_weight='balanced')
clf = CalibratedClassifierCV(base_estimator=estimator, cv=2)

# Quantidade de requisições de rótulos para o oráculo que serão feitas por vez
NUM_QUESTIONS = 3
# categories = [
#     'secretaria de estado de seguranca publica',
#     'secretaria de estado de cultura',
#     'secretaria de estado de fazenda planejamento orcamento e gestao',
#     'casa civil',
#     'secretaria de estado de obras e infraestrutura',
#     'secretaria de estado de educacao',
#     'defensoria publica do distrito federal',
#     'secretaria de estado de saude',
#     'tribunal de contas do distrito federal',
#     'secretaria de estado de desenvolvimento urbano e habitacao',
#     'poder legislativo',
#     'secretaria de estado de justica e cidadania',
#     'secretaria de estado de transporte e mobilidade',
#     'controladoria geral do distrito federal',
#     'poder executivo',
#     'secretaria de estado de agricultura, abastecimento e desenvolvimento rural',
#     'secretaria de estado de economia desenvolvimento, inovacao, ciencia e tecnologia',
#     'secretaria de estado de desenvolvimento economico',
#     'secretaria de estado do meio ambiente']

PATH_TRAIN = "dodftrain.csv"
ENCODING = 'utf-8'
result_x = []
result_y = []

df = pd.read_csv(PATH_TRAIN,encoding = ENCODING,header = 0)
df['label'] = df['label'].map(lambda com: cleanText(com))
df['text'] = df['text'].map(lambda com: cleanText(com))

categories = df.label.unique()

"""Divisao do dataset entre informacoes de treinamento e teste:"""

# df_test = df.sample(frac = 0.33, random_state = 1)
df_test = df.sample(n=230, random_state = 1)

df_train = df.drop(index = df_test.index)

df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

MAX_SIZE = df_train.label.size

"""Cria o dataframe com os exemplos rotulados:

*    Seleciona um exemplo para cada rotulo
"""

df_labeled = pd.DataFrame()

for category in categories:
    df_labeled = df_labeled.append( df_train[df_train.label==category][0:1], ignore_index=True )
    df_train.drop(index = df_train[df_train.label==category][0:1].index, inplace=True)

df_unlabeled = df_train

df_train = df_labeled

# Active learning : loop

while True:

    y_train = df_train.label
    y_test = df_test.label

    df_unlabeled = df_unlabeled.reset_index(drop=True)

    vectorizer = TfidfVectorizer(encoding= ENCODING, use_idf=True, norm='l2', binary=False, sublinear_tf=True,min_df=0.0001, max_df=1.0, ngram_range=(1, 3), analyzer='word', stop_words=None)

    X_train = vectorizer.fit_transform(df_train.text)
    X_test = vectorizer.transform(df_test.text)
    X_unlabeled = vectorizer.transform(df_unlabeled.text)

    df_unified = df_train.append(df_unlabeled)
    X_unified  = vectorizer.transform(df_unified.text)

    question_samples = benchmarkBvSB()
    result_x.append(clfTest())
    result_y.append(df_train.label.size)

    if (df_train.label.size < MAX_SIZE - (len(categories) * NUM_QUESTIONS + 1)) and ((len(result_x) < 2) or ( (result_x[-1] - result_x[-2] > -1) or (result_x[-1] < result_x[-2]) )):
        insert = {'label':[], 'text':[]}
        cont = 0
        for i in question_samples:
            try:
                insert["label"].insert(cont, df_unlabeled.label[i])
                insert["text"].insert(cont, df_unlabeled.text[i])
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
        plt.axis([0, df_train.label.size, 0.3, 1.0])
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

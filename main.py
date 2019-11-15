" Classificadores Implementados "
from Aprendizagem.ZeroR import ZeroR
from Aprendizagem.OneR import OneR
from Aprendizagem.OneRProb import OneRProb
from Aprendizagem.CentroidClassifier import CentroidClassifier
from Aprendizagem.OneRCentroid import OneRCentroid

" Classificadores do scikit-learn "
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

" Ferramentas e bases de dados do scikit-learn "
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from TrabalhoIA import NFoldsTrainTest
import seaborn as sea
from matplotlib import pyplot as plt
import pandas as pd

bases = {
    "iris" : datasets.load_iris(),
    "digits" : datasets.load_digits(),
    "wine" : datasets.load_wine(),
    "breast cancer" : datasets.load_breast_cancer()
}

classificadoresSemHiperparam = {
    # "ZeroR" : (ZeroR, {"discretizar" : False} ),
    # "OneR" : (OneR, {"discretizar" : True} ),
    # "OneR Probabilistico" : (OneRProb, {"discretizar" : True} ),
    "Cassificador Centroide" : (CentroidClassifier, {"discretizar" : False} ),
    # "Centroide OneR" : (OneRCentroid, {"discretizar" : True} ),
    # "Naive Bayes" : (GaussianNB, {"discretizar" : False} )
}

classificadoresComHiperparam = {
    "knn" : (KNeighborsClassifier, {"n_neighbors" : [1, 3, 5, 7, 10] } ),
    "Arvore de Decisao" : (DecisionTreeClassifier, {"max_depth" : [None, 3, 5, 10] } ),
    "Rede Neural" : (MLPClassifier, {"max_iter" : [50, 100, 200], "hidden_layer_sizes" : [(15,)] } ),
    "Floresta Aleatoria" : (RandomForestClassifier, {"n_estimators" : [10, 20, 50, 100] } )
}

treinamento = dict()
for key, base in bases.items():
    treinamento[key] = dict()
    treinamento[key]["Treinador"] = NFoldsTrainTest(base)
    treinamento[key]["Acuracia"] = dict()
    treinamento[key]["Resultados"] = dict()
    for ckey, classificador in classificadoresSemHiperparam.items():
        aux = treinamento[key]["Treinador"].traintest(classificador[0], **classificador[1])
        treinamento[key]["Resultados"][ckey] = aux[2]
        treinamento[key]["Acuracia"][ckey] = aux[0], aux[1]
        aux = pd.Series(treinamento[key]["Acuracia"][ckey][0])
        print("Scores dos Resultados de", ckey, "na base", key, ":", treinamento[key]["Acuracia"][ckey])
        print("Media das Acuracias", aux.mean())
        sea.boxplot(data = treinamento[key]["Resultados"][ckey])
        plt.title("Resultados de " + ckey + " na base " + key + " em cada Fold")
        plt.ylabel("Classes")
        plt.xlabel("Folds")
        plt.savefig("Resultados/Boxplot/folds-" + ckey.replace(" ", "-") + "-" + key.replace(" ", "-") + ".png")
        plt.clf()
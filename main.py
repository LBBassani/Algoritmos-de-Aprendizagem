import warnings
def warn(*args, **kargs):
    pass
warnings.warn = warn

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

# Bases de dados para o trabalho
bases = {
    "iris" : datasets.load_iris(),
    "digits" : datasets.load_digits(),
    "wine" : datasets.load_wine(),
    "breast cancer" : datasets.load_breast_cancer()
}

# Classificadores para realização do experimento
classificadores = {
    "ZeroR" : (ZeroR, {"discretizar" : False} ),
    "OneR" : (OneR, {"discretizar" : True} ),
    "OneR Probabilistico" : (OneRProb, {"discretizar" : True} ),
    "Cassificador Centroide" : (CentroidClassifier, {"discretizar" : False} ),
    "Centroide OneR" : (OneRCentroid, {"discretizar" : True} ),
    "Naive Bayes" : (GaussianNB, {"discretizar" : False} ),
    "knn" : (KNeighborsClassifier, {"hiperparametros" : {"n_neighbors" : [1, 3, 5, 7, 10] } } ),
    "Arvore de Decisão" : (DecisionTreeClassifier, {"hiperparametros" : {"max_depth" : [None, 3, 5, 10] } } ),
    "Rede Neural" : (MLPClassifier, {"hiperparametros" : {"max_iter" : [50, 100, 200], "hidden_layer_sizes" : [(15,)] } } ),
    "Floresta Aleatoria" : (RandomForestClassifier, {"hiperparametros" : {"n_estimators" : [10, 20, 50, 100] } } )
}


# Impressão dos boxplot de cada experimento mostrando os 10 folds de cada classificador em cada base
treinamento = dict()
for key, base in bases.items():
    treinamento[key] = dict()
    treinamento[key]["Treinador"] = NFoldsTrainTest(base)
    treinamento[key]["Acuracia"] = dict()
    treinamento[key]["Resultados"] = dict()
    for ckey, classificador in classificadores.items():
        aux = treinamento[key]["Treinador"].traintest(classificador[0], **classificador[1])
        treinamento[key]["Resultados"][ckey] = aux[2]
        treinamento[key]["Acuracia"][ckey] = aux[0]

# Impressão dos boxlplots das acurácias de cada classificador
for key, base in treinamento.items():
    resultadosBase = list()
    i = 1
    legendas = list()
    for ckey, resul in base["Acuracia"].items():
        resultadosBase.append(pd.DataFrame(resul).assign(Algoritmo = i))
        legendas.append(str(i) + " - " + ckey)
        i += 1
    resultadosBase = pd.concat(resultadosBase)
    resultadosBase = pd.melt(resultadosBase, id_vars=["Algoritmo"], value_name = "Resultados")
    sea.boxplot(x = "Algoritmo", y = "Resultados", data = resultadosBase)
    plt.title("Acurácia dos Experimentos na Base " + key)
    plt.legend(legendas, handlelength=0, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig("Resultados/Boxplot/acuracia-" + key.replace(" ", "-") + ".png", bbox_inches="tight")
    plt.clf()

# Arquivos para impressão das tabelas de resultados
for key, _ in bases.items():
    with open("Resultados/Tabelas/" + key.replace(" ", "-") + ".result", "w") as fp:
        fp.write("Resultados de " + key + "\n\\\\Algoritmo & Média das Acurácias & Desvio Padrão das Acurácias")
        for ckey, resul in treinamento[key]["Acuracia"].items():
            fp.write("\n\\\\" + str(ckey) + " & " + str(pd.Series(resul).mean()) + " & " + str(pd.Series(resul).std()))

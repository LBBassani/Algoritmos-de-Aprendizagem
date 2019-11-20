from sklearn.model_selection import KFold, GridSearchCV
from sklearn import preprocessing
import pandas as pd

class InvalidNFoldsException(Exception):
    pass

class NFoldsTrainTest(object):

    def __init__(self, base, n_folds = 10, n_cycles = 4, random_state = 0, shuffle = False):
        if len(base.data)//n_folds < 1:
            raise InvalidNFoldsException
        self.__base = base
        self.__nfolds = n_folds
        self.__ncycles = n_cycles
        self.__cv = KFold(n_splits = n_folds, random_state = random_state, shuffle = shuffle)
    
    def traintest(self, classifier, discretizar = False, hiperparametros = None):
        if hiperparametros is not None:
            classificador = GridSearchCV(classifier(),hiperparametros, cv = self.__ncycles, verbose=0)
        else:
            classificador = classifier()
        scores = list()
        desvios = list()
        resultados = list()
        X = self.__base.data
        y = self.__base.target
        for train_index, test_index in self.__cv.split(X):
            X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
            if discretizar == True:
                enc = preprocessing.KBinsDiscretizer(n_bins=([len(X[0])]*len(X[0])), encode="ordinal", strategy="kmeans").fit(X_train)
                X_bin_train = enc.transform(X_train)
                X_bin_test = enc.transform(X_test)
            else:
                X_bin_train = X_train
                X_bin_test = X_test
            if hiperparametros is not None: # Realiza treinamento, validação e teste com hiperparâmetros
                classificador.fit(X_train, y_train)
                print(classificador.best_params_, classificador.best_score_)
            else: # Realiza treinamento e teste sem hiperparâmetros
                classificador.fit(X_bin_train, y_train)
            y_pred = classificador.predict(X_bin_test)
            resultados.append(y_pred)
            desvios.append(pd.Series(y_pred).std())
            scores.append(classificador.score(X_bin_test, y_test))
        return scores, desvios, resultados
import warnings
def warn(*args, **kargs):
    pass
warnings.warn = warn

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from .OneR import OneR
import random as r

class OneRProb(OneR, BaseEstimator, ClassifierMixin):
    def __init__(self):
        OneR.__init__(self)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        best_predictor = -1
        best_error = len(self.X_)
        table = self._OneR__calculate_erros()
        best_predictor, best_error = self.__roleta(table)
        print("Melhor preditor", best_predictor)
        # Guarda o melhor preditor e a melhor regra no formato [(condição, resposta)]
        self.best_ = best_predictor
        self.regra_ = table[best_predictor][2]
        print("Regras", self.regra_)
        return self

    def __roleta(self, estados):
        """ primeiro passo : definir as faixas de sobrevivência
                Como :  calcular as probabilidades de cada um sobreviver (aptidao/sum(aptidoes))
                        calcular a faixa de sobrevivência
        """
        total = sum(list(map(lambda x: 1/x[1], estados)))
        porcentagens = list(map(lambda x: (x, (1/x[1])/total),estados))

        print("Chances de Escolher cada Preditor:", porcentagens)
        faixaSobrevivencia = list()
        limiteInf = 0
        for e in porcentagens:
            faixaSobrevivencia.append((limiteInf, limiteInf + e[1], e[0]))
            limiteInf = limiteInf + e[1]
        
        """ segundo passo : escolher o sobrevivente 
                Como :  gerar um número aleatório
                        descobrir em qual faixa de sobrevivência ele se encontra
        """
        n = r.uniform(0,1)
        for e in faixaSobrevivencia:
            if n >= e[0] and n < e[1]:
                return e[2][0], e[2][1]
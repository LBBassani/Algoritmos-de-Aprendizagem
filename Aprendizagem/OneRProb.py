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
        best_error = len(self.X_[0])
        table = list()
        for i in range(0, len(self.X_[0])):
            predictor_table = list()
            x = self._OneR__constructArray(self.X_, [i])     # Escolhe o preditor
            prediction = tuple(zip(x, self.y_))         # Faz tuplas com (valor do preditor, classe)
            erros = list()
            for valor in unique_labels(x):
                classe_mais_frequente = -1
                frequencia_mais_frequente = 0
                line = list()
                for classe in self.classes_:
                    # Encontra quantos registros tem o valor e a classe
                    q = len(list(filter(lambda x: x[1] == classe and x[0] == valor, prediction)))
                    if q >= frequencia_mais_frequente:
                        frequencia_mais_frequente = q
                        classe_mais_frequente = classe 
                    line.append((classe, q))
                erro = sum(list(map(lambda x: x[1],list(filter(lambda x: x[0] != classe_mais_frequente, line)))))
                erros.append(erro)
                predictor_table.append((valor, classe_mais_frequente))
            erro = sum(erros)
            print("preditor", i, "erros", erro, "Tabela de Predição", predictor_table)
            table.append((i, erro, predictor_table))
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
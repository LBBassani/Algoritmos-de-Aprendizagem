from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import numpy as np

class OneR(BaseEstimator, ClassifierMixin):
    """ OneR Algorithm		
            For each predictor,
                For each value of that predictor, make a rule as follows;
                    Count how often each value of target (class) appears            
                    Find the most frequent class                                    
                    Make the rule assign that class to this value of the predictor  
                Calculate the total error of the rules of each predictor            
            Choose the predictor with the smallest total error.                     
    """
    def __constructArray(self, matriz, colunas):
        resp = list()
        matriz = np.array(matriz).T
        for c in colunas:
            resp.append(matriz[c])
        return np.array(resp).T

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
            x = self.__constructArray(self.X_, [i])     # Escolhe o preditor
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
            # Encontra o preditor com o menor erro
            if erro <= best_error:
                best_error = erro
                best_predictor = i
        print("Melhor preditor", best_predictor)
        # Guarda o melhor preditor e a melhor regra no formato [(condição, resposta)]
        self.best_ = best_predictor
        self.regra_ = table[best_predictor][2]
        print("Regras", self.regra_)
        return self

    def predict(self, X):
        resp = list()
        for x in self.__constructArray(X, [self.best_]):
            for regra in self.regra_:
                if x == regra[0]:
                    resp.append(regra[1])
        return resp
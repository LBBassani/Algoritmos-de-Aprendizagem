import warnings
def warn(*args, **kargs):
    pass
warnings.warn = warn

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.spatial import distance

class CentroidClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = list(X)
        self.y_ = y
        self.centroides_ = list()
        for c in self.classes_:
            classe = list()
            for i in range(len(self.X_)):
                if self.y_[i] == c:
                    classe.append(self.X_[i])
            centroid = [0]*len(self.X_[0])
            for componente in classe:
                for i in range(len(componente)):
                    centroid[i] += componente[i]
            centroid = list(map(lambda x: x/len(classe), centroid))
            self.centroides_.append((centroid, c))
        return self
    
    def predict(self, X):
        resp = list()
        for x in X:
            distancias = list()
            for centroid, c in self.centroides_:
                dist = distance.euclidean(centroid, x)
                distancias.append((dist, c))
            best_dist, best_class = min(distancias)
            resp.append(best_class)
        return resp
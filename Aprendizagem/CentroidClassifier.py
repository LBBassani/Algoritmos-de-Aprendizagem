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
        self.X_ = X
        self.y_ = y
        self.centroides_ = list()
        for c in self.classes_:
            classe = list(filter(lambda x: self.y_[self.X_.index(x)] == c, self.X_ ))
            centroid = [0]*len(self.X_[0])
            for componente in classe:
                for i in range(len(componente)):
                    centroid[i] += componente[i]
            centroid = list(map(lambda x: x/len(classe), centroid))
            self.centroides_.append(centroid)
        return self
    
    def predict(self, X):
        resp = list()
        for x in X:
            best_dist = 0
            best_classe = -1
        return resp
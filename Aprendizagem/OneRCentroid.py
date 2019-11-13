import warnings
def warn(*args, **kargs):
    pass
warnings.warn = warn

from .CentroidClassifier import CentroidClassifier
from .OneR import OneR

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.spatial import distance
import numpy as np

class OneRCentroid(OneR, CentroidClassifier, BaseEstimator, ClassifierMixin):
    def __init__(self):
        OneR.__init__(self)
        CentroidClassifier.__init__(self)
    
    def fit(self, X, y):
        self = OneR.fit(self, X, y)
        x = self._OneR__constructArray(X, [self.best_])
        self = CentroidClassifier.fit(self, x, y)
        return self

    def predict(self, X):
        x = self._OneR__constructArray(X, [self.best_])
        return CentroidClassifier.predict(self, x)
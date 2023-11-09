import numpy as np

class LaplaceDistribution:    
    @staticmethod
    def mean_abs_deviation_from_median(x: np.ndarray):
        '''
        Args:
        - x: A numpy array of shape (n_objects, n_features) containing the data
          consisting of num_train samples each of dimension D.
        '''
        ####
        # Do not change the class outside of this block
        # Your code here
        mu = []
        b = []
        n_features = 0
        # Check if the array is 1D or 2D
        if len(x.shape) == 1:
            mu.append(np.median(x))
        else:
            n_features = x.shape[1]
            for i in range(n_features):
                mu.append(np.median(x[: , i]))
        n_objects = x.shape[0]
        for i in range(n_features):
            b.append(1 / n_objects * np.sum(np.abs(x[:, i] - mu[i])))

        return b
        ####

    def __init__(self, features):
        '''
        Args:
            features: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        ####
        # Do not change the class outside of this block
        mu = []
        # Check if the array is 1D or 2D
        if len(features.shape) == 1:
            mu.append(np.median(features))
        else:
            n_features = features.shape[1]
            for i in range(n_features):
                mu.append(np.median(features[:, i]))

        self.loc = mu
        self.scale = LaplaceDistribution.mean_abs_deviation_from_median(features)
        ####


    def logpdf(self, values):
        '''
        Returns logarithm of probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        ####
        # Do not change the class outside of this block
        n_features = values.shape[1]
        logpdf = []
        for i in range(n_features):
            logpdf.append(np.log(1 / (2 * self.scale[i])) - np.abs(values[: , i] - self.loc[i]) / self.scale[i])
        return logpdf
        ####
        
    
    def pdf(self, values):
        '''
        Returns probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        return np.exp(self.logpdf(values))

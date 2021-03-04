from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from copy import deepcopy

df = pd.DataFrame({'country': ['russia', 'germany', 'australia','korea','germany'],
                   'words': ['python', 'is', 'python','cool','me']})
df

var = 'country'
var = 'words'
categories = df[var].unique().tolist()

# add unknown catgory for out of sample levels
df[var] = df[var].astype(CategoricalDtype(categories + ['OTHERS']))

one = pd.get_dummies(df[var], prefix='', prefix_sep='')
one


def onehot(df, oos_token = 'OTHERS'):
   dummy = None
   for z, var in enumerate(df.columns):
        categories = df[var].unique().tolist()
        df[var] = df[var].astype(CategoricalDtype(categories + [oos_token]))    # add unknown catgory for out of sample levels
        one = pd.get_dummies(df[var], prefix='var')
        if z>0:
           dummy = pd.concat([dummy, one], axis=1)
        else:
           dummy = one.copy()    
   return dummy    
#
onehot(df)



from sklearn.base import BaseEstimator, TransformerMixin


class onehot_encoder(TransformerMixin, BaseEstimator):

    def __init__(self, oos_token = 'OTHERS', verbose = True, **kwargs):
        """
        One-hot encoder that handles out-of-sample levels of categorical variables

        Args:
            oos_token (str, optional): [description]. Defaults to 'OTHERS'.
        """
        self.oos_token = oos_token
        self.kwargs = kwargs
        self.verbose = verbose
        if self.verbose : print("One-hot encoding of categorical features")

    def fit(self, X):

        df = deepcopy(X)
        for z, var in enumerate(df.columns):
            categories = df[var].unique().tolist()
            df[var] = df[var].astype(CategoricalDtype(categories + [self.oos_token]))    # add unknown catgory for out of sample levels
            one = pd.get_dummies(df[var], prefix=var, **self.kwargs)
            if z>0:
               dummy = pd.concat([dummy, one], axis=1)
            else:
               dummy = deepcopy(one)
        self.dummy_ = dummy           
        self.columns_ = list(self.dummy_.columns) 
        return self    

    def transform(self, X):
        return self.dummy_

oh = onehot_encoder(prefix_sep='_')
fitted = oh.fit(X = df)
oh.transform(df)
oh.columns_

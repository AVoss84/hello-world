# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 16:43:26 2017

@author: z002ffcz
"""
       
import numpy as np
from pandas import DataFrame
from pandas import concat


# Creates a sequence of integers:
#---------------------------------
def sequence(low,up):
  res = [] ; 
  for i in range(low,up+1):
    #print(i) ;
    res.append(i)
  return(res)  
#-----------------------------------
#sequence(1,10)


# Input must be boolean vector:
#------------------------------------------------------------------------------- -----   
def which(x):
      if (not isinstance(x,list) | ('np' not in str(type(x)))) :
         #exit();
         raise ValueError("Must be an object of type numpy or a list!");
      nx = len(x) ; y = [] ;
      for i in range(0,nx) :
          if(x[i]):
              y.append(i) ;          # here locations are indexed from 0 not from 1
      return(y)
#------------------------------------------------------------------------------------
#inp_bool = np.random.binomial(n,p,size) ; print(inp_bool)       # draw Bernoulli rvs
#which(inp_bool)
#np.nonzero(inp_bool)        # or use simply....


#------------------------------------------------
def updown_Py(myvector, start = None):
    # Initialize some objects:
    myvector = np.matrix(myvector).T         # default is row matrix so transpose it column matrix
    m = myvector.shape[0] ; #print(nrow)
    updown = np.chararray((m, 1), itemsize=4)           # itemsize:  Length of each array element, in number of characters. Default is 1.
    updown[0] = 'stay';
    
    if start is not None:
        #print('Start not None!\n') ; 
        if myvector[0] > start:             # for nested if statements the cursor position serves as {...}
          updown[0] = 'up'
        elif myvector[0] < start:
          updown[0] = 'down'
          
    if m > 1:
        for i in range(0,m-1):
         #print(i)
         if myvector[i+1] > myvector[i]:
           updown[i+1] = "up"
         elif myvector[i+1] < myvector[i]:
           updown[i+1] = "down"
         elif myvector[i+1] == myvector[i]:
           updown[i+1] = "stay"
         
    return updown;
#------------------------------------------------

# Call function:
#----------------  
#c = updown_Py(series, start = -88) #.head()

# Analogue to R's embed() function
#------------------------------------------------------------------------------
def embed(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names                      # assign new column labels
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
#--------------------------------------------------------------------------------

### Use it
values = [x for x in range(10)] ; values
data = embed(values, n_in=2) ;print(data)













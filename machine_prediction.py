import numpy as np
from sklearn import svm, preprocessing
import pandas as pd
import warnings




FEATURES = ['averageD',
            'triSpread',
            'averageSpread',
            'F1_EXT',
            'F2_EXT',
            'F3_EXT',
            'F4_EXT',
            'F5_EXT',]

def build_data_set():
    
    data_df = pd.DataFrame.from_csv('test01_data.csv')

    data_df = data_df.reindex(np.random.permutation(data_df.index))

    X = np.array(data_df[FEATURES].values)
    
    #X = preprocessing.scale(X)
   

    y = (data_df['sign'].values.tolist())
    

    #Z = np.array(data_df[['averageD' , 'triSpread' , 'averageSpread']])

    return X,y

def analysis ():
    
    test_size = 30
    
    X, y = build_data_set()
    print (len(X))
    

    clf = svm.SVC(kernel="linear", C= 1.0)
    clf.fit(X[:-test_size],y[:-test_size])
    #clf.fit(X,y)

    correct_count = 0

    for x in range(1, test_size+1):
            if clf.predict(X[-x])[0] == y[-x]:
                correct_count += 1

    print ('Accuracy:' , (correct_count/test_size) * 100.00)

# prediction code

    data_df = pd.DataFrame.from_csv('new0_data.csv')

    X = np.array(data_df[FEATURES].values)
  
    #X = preprocessing.scale(X)
  
    #Z = np.array(data_df[['averageD' , 'triSpread' , 'averageSpread']])
   
    p = clf.predict(X)
    print(p)


def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()            
    analysis()


    

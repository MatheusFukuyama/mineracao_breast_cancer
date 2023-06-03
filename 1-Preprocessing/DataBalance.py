from imblearn.over_sampling import RandomOverSampler, SMOTE
import pandas as pd
from collections import Counter
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Random Oversampler
# ros = RandomOverSampler(random_state = 32)
# X_ros_res, y_ros_res = ros.fit_resample(X, y)

def main():
    # Load iris data and store in dataframe
    names = ['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad',
             'irradiat']
    features = ['age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast',
                'breast-quad', 'irradiat']
    target = 'Class'
    input_file = '0-Datasets/br-out.data'
    df = pd.read_csv(input_file,  # Nome do arquivo com dados
                     names=names)

    teste, target_names = pd.factorize(df.loc[:, target].values)

    # identify all categorical variables
    cat_columns = df.select_dtypes(['object']).columns

    # convert all categorical variables to numeric
    df[cat_columns] = df[cat_columns].apply(lambda x: pd.factorize(x)[0])

    # Separating out the features
    x = df.loc[:, features].values
    # print(x, "aqui")
   
    print("Total samples: {}".format(x.shape[0]))

    # Separating out the target
    y = df.loc[:, target].values

    print('Original dataset shape %s' % Counter(y))

    SMOTE
    smote = SMOTE(random_state = 32)
    x_smote_res, y_smote_res = smote.fit_resample(x, y) 

    print('Original dataset shape %s' % Counter(y_smote_res))

    # print(y)

    # Get test confusion matrix
 

if __name__ == "__main__":
    main()  
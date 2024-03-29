import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def main():
    # Faz a leitura do arquivo
    names = ['Class','age','menopause','tumor-size','inv-nodes','node-caps','deg-malig','breast','breast-quad','irradiat'] 
    features =  ['age','menopause','tumor-size','inv-nodes','node-caps','deg-malig','breast','breast-quad','irradiat'] 
    target = 'Class'
    input_file = '../0-Datasets/br-out.data'
    output_file = '../0-Datasets/normalizedFile.data'
    df = pd.read_csv(input_file,         # Nome do arquivo com dados
                    names = names)      
   # Nome das colunas                      
    ShowInformationDataFrame(df,"Dataframe original")

    #identify all categorical variables
    cat_columns = df.select_dtypes(['object']).columns

    # ShowInformationDataFrame(a,"Dataframe ordenado")

    # convert all categorical variables to numeric
    df[cat_columns] = df[cat_columns].apply(lambda x: pd.factorize(x)[0])
    df.to_csv(output_file, header=False, index=False)


    # Separating out the features
    x = df.loc[:, features].values
    
    # Separating out the target
    y = df.loc[:,[target]].values

    # Z-score normalization
    # x_zcore = StandardScaler().fit_transform(x)
    # normalized1Df = pd.DataFrame(data = x_zcore, columns = features)
    # normalized1Df = pd.concat([normalized1Df, df[[target]]], axis = 1)
    # ShowInformationDataFrame(normalized1Df,"Dataframe Z-Score Normalized")

    # Mix-Max normalization
    x_minmax = MinMaxScaler().fit_transform(x)
    normalized2Df = pd.DataFrame(data = x_minmax, columns = features)
    normalized2Df = pd.concat([normalized2Df, df[[target]]], axis = 1)
    ShowInformationDataFrame(normalized2Df,"Dataframe Min-Max Normalized")

    #  df.to_csv("../0-Datasets/DataNormalized.data", header=False, index=False)  
    


def ShowInformationDataFrame(df, message=""):
    print(message+"\n")
    print(df.info())
    print(df.describe())
    print(df.head(10))
    print("\n") 



if __name__ == "__main__":
    main()

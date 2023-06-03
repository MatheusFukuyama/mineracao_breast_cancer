from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import StandardScaler

def main():
    names = ['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad',
             'irradiat']
    features = ['age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast',
                'breast-quad', 'irradiat']
    target = 'Class'
    input_file = '0-Datasets/br-out.data'
    df = pd.read_csv(input_file,  # Nome do arquivo com dados
                     names=names)

    # identify all categorical variables
    cat_columns = df.select_dtypes(['object']).columns

    # convert all categorical variables to numeric
    df[cat_columns] = df[cat_columns].apply(lambda x: pd.factorize(x)[0])

    # Separating out the features
    x = df.loc[:, features].values

    # Separating out the target
    y = df.loc[:, [target]].values

    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    normalizedDf = pd.DataFrame(data=x, columns=features)
    print(normalizedDf)

    print(x)
    print(y)

    """
    # Split the data - 75% train, 25% test
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)    
    iris = load_iris()
    print(iris.data)
    print(iris.target)
    X = iris.data
    y = iris.target
    """

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    clf = DecisionTreeClassifier(max_leaf_nodes=3)
    print(X_train)
    clf.fit(X_train, y_train)
    tree.plot_tree(clf)
    plt.show()

    predictions = clf.predict(X_test)
    print(predictions)

    result = clf.score(X_test, y_test)
    print('Acuraccy:')
    print(result)


if __name__ == "__main__":
    main()
# demo for classification

import numpy as np
import pandas as pd

# dummy features
from sklearn.impute import SimpleImputer

# split dataset
from sklearn.model_selection import train_test_split

# feature scaling
from sklearn.preprocessing import StandardScaler

# from catboost import CatBoostClassifier
# auto tune
from catboost import CatBoostClassifier

from sklearn.metrics import confusion_matrix, accuracy_score

# k_fold
from sklearn.model_selection import cross_val_score


if __name__ == '__main__':
    # import dataset
    dataset = pd.read_csv('train.tsv', sep='\t')
    print(dataset)

    # cut off unnecessary info
    X = dataset.iloc[:, 1:-1].values
    y = dataset.iloc[:, -1].values

    # dummy features if need
    impute = SimpleImputer(missing_values=np.nan, strategy='mean')
    impute.fit(X)
    X = impute.transform(X)

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # feature scaling if need
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # fit dataset with catboost
    classifier = CatBoostClassifier()
    classifier.fit(X_train, y_train)

    # import test for submitting dataset
    test = dataset = pd.read_csv('test.tsv', sep='\t')

    # cut off id for exporting
    num = test.iloc[:, 0].values

    # these values go to predict
    test = test.values
    test = sc.transform(test[:, 1:])

    # prediction
    y_pred = classifier.predict(test)

    # for exporting
    y_export = y_pred.reshape(len(y_pred), 1)
    num = num.reshape(len(num), 1)
    export_data = np.concatenate((num, y_export), axis=1)

    export_data = export_data.astype(int)

    # check accuracy
    cm = confusion_matrix(y_test, classifier.predict(X_test))
    print(cm)
    print(export_data)
    print(accuracy_score(y_true=y_test, y_pred=classifier.predict(X_test)))

    # K-fold
    # acccuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
    # print('The mean of accuracy is {:.2f} %'.format(acccuracies.mean()))
    # print('The SD of accuracy is {:.2f} %'.format(acccuracies.std()))

    # if the accuracy is okay, ready to export
    pd.DataFrame(export_data).to_csv("result.csv", index=False)

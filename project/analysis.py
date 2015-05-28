#!/usr/bin/python3
# Analyze Titanic survivor data

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm, tree, naive_bayes, ensemble
from sklearn.cross_validation import cross_val_score

def build_nb_classifier(X, y):
    """ Construct a classifier Naive Bayes

    :param X: Example set
    :param y: Prediction
    """
    nb = naive_bayes.GaussianNB()
    return nb.fit(X, y)


def build_svm(X, y):
    """ Construct a classifier using a Support Vectore Machine

    :param X: Example set
    :param y: Prediction
    """
    svc = svm.LinearSVC()
    return svc.fit(X, y)


def build_decision_tree(X, y):
    """ Construct a classifier using a decision tree

    :param X: Example set
    :param y: Prediction
    """
    decision_tree = tree.DecisionTreeClassifier()
    return decision_tree.fit(X, y)


def build_random_forest(X, y):
    """ Construct a classifier using a random forest

    :param X: Example set
    :param y: Prediction
    """
    forest = ensemble.RandomForestClassifier()
    return forest.fit(X, y)


def munge_data(df):
    """ Work the data into an appropriate form for analysis

    A number of adjustments to the data need to be made to prepare it
    for the classifiers. A new column 'Gender' is added, which is set
    to 1 if male and 0 if female. A new column 'AgeFixed' is added, which
    currently substitutes the mean Age: in the future we wish to make this
    substitution more robust.

    :param df: Pandas Data Frame
    :returns: Data Frame after data adjustments
    """
    # Create a new Gender column from Sex with the mapping female -> 0, male -> 1
    df['Gender'] = df['Sex'].map(lambda x: 1 if x == 'male' else 0)

    # Create a AgeFixed column, substituting the mean Age for msising values
    average_age = df['Age'].mean()
    age_fix = lambda x: average_age
    df['AgeFixed'] = df['Age'].map(age_fix)

    return df


def main():
    # Set number of folds for cross-validation
    k = 10

    # Load Titanic passenger csv into a data frame
    df = pd.read_csv('train.csv', header=0)
    test_set = pd.read_csv('test.csv', header=0)

    # Prepare data format
    df = munge_data(df)
    test_set = munge_data(test_set)
    target = df['Survived']
    survivors = df[(df['Survived'] == 1)]
    perishers = df[(df['Survived'] == 0)]

    #print("Survival Rate: %.2f" % df["Survived"].mean())
    print("Average Age: %.2f" % df["Age"].mean())
    print("Average Age of Survivor: %.2f" % survivors["Age"].mean())
    print("Average Age of Perisher: %.2f" % perishers["Age"].mean())
    print(df.head())

    # Finally, reduce our feature set
    features = ['AgeFixed', 'Gender', 'Pclass']
    df = df[features]
    test_set = test_set[features]

    # Build classifiers
    classifiers = {
            'nb': build_nb_classifier(df, target),
            'svm': build_svm(df, target),
            'forest': build_random_forest(df, target),
            'tree': build_decision_tree(df, target)
    }

    # Ferform k-fold cross validation and render results
    for key in classifiers.keys():
        print("Executing " + str(k) + " folds cross-validation for " + key)
        scores = cross_val_score(classifiers[key], df, target, cv=k)
        print(scores)


if __name__ == '__main__':
    main()

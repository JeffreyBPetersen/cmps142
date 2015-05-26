#!/usr/bin/python3
# Analyze Titanic survivor data

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm, tree, naive_bayes, ensemble

def build_nb_classifier(df):
    """ Construct a classifier Naive Bayes

    :param df: Pandas Data Frame
    """
    # Separate observations and target
    features = ['AgeFixed', 'Gender', 'Pclass']
    X = df[features]
    y = df['Survived']
    nb = naive_bayes.GaussianNB()
    return nb.fit(X, y)


def build_svm(df):
    """ Construct a classifier using a Support Vectore Machine

    :param df: Pandas Data Frame
    """
    # Separate observations and target
    features = ['AgeFixed', 'Gender', 'Pclass']
    X = df[features]
    y = df['Survived']
    svc = svm.LinearSVC()
    return svc.fit(X, y)


def build_decision_tree(df):
    """ Construct a classifier using a decision tree

    :param df: Pandas Data Frame
    """
    # Separate observations and target
    features = ['AgeFixed', 'Gender', 'Pclass']
    X = df[features]
    y = df['Survived']
    decision_tree = tree.DecisionTreeClassifier()
    return decision_tree.fit(X, y)


def build_random_forest(df):
    """ Construct a classifier using a random forest

    :param df: Pandas Data Frame
    """
    # Separate observations and target
    features = ['AgeFixed', 'Gender', 'Pclass']
    X = df[features]
    y = df['Survived']
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
    print(df[['AgeFixed', 'Gender', 'Pclass']])
    return df


def main():
    # Load Titanic passenger csv into a data frame
    df = pd.read_csv('train.csv', header=0)
    test_set = pd.read_csv('test.csv', header=0)
    df = munge_data(df)
    survivors = df[(df['Survived'] == 1)]
    perishers = df[(df['Survived'] == 0)]

    print("Survival Rate: %.2f" % df["Survived"].mean())
    print("Average Age: %.2f" % df["Age"].mean())
    print("Average Age of Survivor: %.2f" % survivors["Age"].mean())
    print("Average Age of Perisher: %.2f" % perishers["Age"].mean())
    print(df.head())

    # Build classifiers
    classifiers = {
            'nb': build_nb_classifier(df),
            'svm': build_svm(df),
            'forest': build_random_forest(df),
            'tree': build_decision_tree(df)
    }


if __name__ == '__main__':
    main()

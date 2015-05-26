#!/usr/bin/python3
# Analyze Titanic survivor data

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree

def build_nb_classifier(df):
    """ Construct a classifier Naive Bayes

    :param df: Pandas Data Frame
    """
    pass

def build_svm(df):
    """ Construct a classifier using a Support Vectore Machine

    :param df: Pandas Data Frame
    """
    pass

def build_decision_tree(df):
    """ Construct a classifier using a decision tree

    :param df: Pandas Data Frame
    """
    # Separate observations and target
    X = df.copy()
    del X['Survived']
    y = df['Survived']
    decision_tree = tree.DecisionTreeClassifier()
    # TODO finish converting data into expected format
    #decision_tree = decision_tree.fit(X, y)

def main():
    # Load Titanic passenger csv into a data frame
    df = pd.read_csv('train.csv', header=0)
    survivors = df[(df['Survived'] == 1)]
    perishers = df[(df['Survived'] == 0)]
    print("Survival Rate: %.2f" % df["Survived"].mean())
    print("Average Age: %.2f" % df["Age"].mean())
    print("Average Age of Survivor: %.2f" % survivors["Age"].mean())
    print("Average Age of Perisher: %.2f" % perishers["Age"].mean())
    print(df.head())

    build_nb_classifier(df);
    build_svm(df);
    build_decision_tree(df);


if __name__ == '__main__':
    main()

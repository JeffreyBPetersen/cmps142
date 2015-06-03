#!/usr/bin/python3
# Analyze Titanic survivor data

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm, tree, naive_bayes, ensemble
from sklearn.cross_validation import cross_val_predict
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

def build_nb_classifier(X, y):
    """ Construct a classifier Naive Bayes

    :param X: Example set
    :param y: Prediction
    """
    nb = naive_bayes.GaussianNB()
    nb.fit(X, y)
    return nb


def build_svm(X, y, gs_params = None, k = 10):
    """ Construct a classifier using a Support Vectore Machine

    :param X: Example set
    :param y: Prediction
    :param gs_params: None, or array of grid search parameters
    :param k: Folds for cross validation
    """
    if gs_params:
        svc = GridSearchCV(svm.SVC(), gs_params, cv = k) #, n_jobs = -1)
        svc = svc.fit(X, y)
        print("Resulting Grid Search Params: ", svc.best_params_)
    else:
        svc = svm.LinearSVC()
        svc = svc.fit(X, y)
    return svc


def build_decision_tree(X, y, gs_params = None, k = 10):
    """ Construct a classifier using a decision tree

    :param X: Example set
    :param y: Prediction
    :param gs_params: None, or array of grid search parameters
    :param k: Folds for cross validation
    """
    decision_tree = tree.DecisionTreeClassifier()
    if gs_params:
        decision_tree = GridSearchCV(decision_tree, gs_params, cv = k) #, n_jobs = -1)
        decision_tree = decision_tree.fit(X, y)
        print("Resulting Grid Search Params: ", decision_tree.best_params_)
    else:
        decision_tree.fit(X, y)
    return decision_tree


def build_random_forest(X, y, gs_params = None, k = 10):
    """ Construct a classifier using a random forest

    :param X: Example set
    :param y: Prediction
    :param gs_params: None, or array of grid search parameters
    :param k: Folds for cross validation
    """
    forest = ensemble.RandomForestClassifier(n_estimators = 100)
    if gs_params:
        forest = GridSearchCV(forest, gs_params, cv = k) #, n_jobs = -1)
        forest = forest.fit(X, y)
        print("Resulting Grid Search Params: ", forest.best_params_)
    else:
        forest = forest.fit(X, y)
    return forest


def split_dataset(X, y):
    """ Split a DataFrame into testing and training sets

    :param X: Example set
    :param y: Prediction
    """
    train_X, test_X, train_y, test_y = train_test_split(X, y)
    train = {
            "examples": train_X,
            "target": train_y
            }
    test = {
            "examples": train_X,
            "target": train_y
            }
    return (train, test)


def cross_validate_predict(classifiers, X, y, k):
    """  Perform k-fold cross validation and render prediction function

    :param classifiers: Dictionary of classifiers, indexed by name
    :param X: Example set
    :param y: Instance label
    :param k: Number of folds
    :returns: List of predictions
    """
    predictions = {}
    for key in classifiers.keys():
        prediction = cross_val_predict(classifiers[key], X, y, cv=k, n_jobs=-1)
        predictions[key] = prediction
        
    return predictions

def cross_validate_classifiers(classifiers, X, y, k):
    """  Perform k-fold cross validation and render accuracies

    :param classifiers: Dictionary of classifiers, indexed by name
    :param X: Example set
    :param y: Prediction
    :param k: Number of folds
    """
    for key in classifiers.keys():
        print("\nExecuting " + str(k) + " folds cross-validation for " + key)
        scores = cross_val_score(classifiers[key], X, y, cv=k, n_jobs=-1)
        print(scores)
        max = scores.max()
        max_index = np.where(scores == max)[0][0] + 1
        print("Mean: " + str(np.mean(scores)) + ", Max: " + str(max) +
                " (" + str(max_index) + " folds)")

# TODO - Create more feature engineering methods
 
def pclass_avg_age_fill(df):
    for i in range(1,4):
        pclass_df = df.loc[df['Pclass'] == i]
        pclass_avg = pclass_df['Age'].mean()
        age_fix = lambda x: pclass_avg
        df['AgeFixed'] = df['Age'].map(age_fix)
    
    return df
    
def avg_age_fill(df):
    """ Fill in missing values in the age column with the average age over the entire data set
    
    :param df: Pandas Data Frame
    :returns: Data Frame with no missing values in the age column 
    """
    # Create a AgeFixed column, substituting the mean Age for missing values
    average_age = df['Age'].mean()
    age_fix = lambda x: average_age
    df['AgeFixed'] = df['Age'].map(age_fix)
    
    return df

def munge_data(df, fill_method):
    """ Work the data into an appropriate form for analysis

    A number of adjustments to the data need to be made to prepare it
    for the classifiers. A new column 'Gender' is added, which is set
    to 1 if male and 0 if female. A new column 'AgeFixed' is added, which
    currently substitutes the mean Age: in the future we wish to make this
    substitution more robust.

    :param df: Pandas Data Frame
    :param fill_method: Method used to fill in missing data for age
                        (in the future this may be turned into a dictionary to allow for any feature to be filled)
    :returns: Data Frame after data adjustments
    """
    # Create a new Gender column from Sex with the mapping female -> 0, male -> 1
    df['Gender'] = df['Sex'].map(lambda x: 1 if x == 'male' else 0)

    # Create a new Port column to represent the port Embarked from
    df['Port'] = df['Embarked'].map({'C': 1, 'Q': 2, 'S': 3, None: 0})
    
    fill_methods = {
        'total_avg': avg_age_fill(df),
        'pclass_avg': pclass_avg_age_fill(df)
    }
    
    df = fill_methods[fill_method]

    return df


def main(args):
    parser = argparse.ArgumentParser(description='Titanic survival prediction analysis tool')
    parser.add_argument('-k', '--folds', help='Number of folds for cross validation',
            required=False, default=10, type=int)
    parser.add_argument('-g', '--gridsearch', help='Perform grid search [ SVM ]',
            required=False, default=False, action='store_true')
    parser.add_argument('--agefill', help='Method to use in order to fill in missing values in the age column',
            required=False, default='total_avg')
    args = parser.parse_args(args)

    # Load Titanic passenger csv into a data frame
    df = pd.read_csv('train.csv', header=0)
    # Load final test set for predictions
    test_set = pd.read_csv('test.csv', header=0)

    # Prepare data format
    df = munge_data(df, args.agefill)
    test_set = munge_data(test_set, args.agefill)
    target = df['Survived']
    survivors = df[(df['Survived'] == 1)]
    perishers = df[(df['Survived'] == 0)]

    #print("Survival Rate: %.2f" % df["Survived"].mean())
    print("Average Age: %.2f" % df["Age"].mean())
    print("Average Age of Survivor: %.2f" % survivors["Age"].mean())
    print("Average Age of Perisher: %.2f" % perishers["Age"].mean())
    print(df.head())

    # Finally, reduce our feature set
    features = ['AgeFixed', 'Gender', 'Pclass', 'Port']
    df = df[features]
    test_set = test_set[features]

    # Define some grid search parameters
    cost_search = [1, 5, 10, 15]
    gamma_search = [0.1, 0.01]
    if args.gridsearch:
        svm_params = [{'C': cost_search, 'kernel': ['rbf'], 'gamma': gamma_search},
                     {'C': cost_search, 'kernel': ['linear']}]
        forest_params = [{'bootstrap': [True, False], 'max_depth': [2, 3, 4, None], 'max_features' : [2, 3, None]}]
        tree_params = [{'max_depth': [2, 3, 4, None], 'max_features' : [2, 3, None]}]
    else:
        svm_params = None
        forest_params = None
        tree_params = None

    # Build classifiers
    classifiers = {
            'nb': build_nb_classifier(df, target),
            'svm': build_svm(df, target, svm_params, args.folds),
            'forest': build_random_forest(df, target, forest_params, args.folds),
            'tree': build_decision_tree(df, target, tree_params, args.folds)
    }

    # Cross Validate k-fold
    # TODO - add scoring
    post_train, post_test = split_dataset(df, target)
    post_train_target = post_train['target']
    post_train = post_train['examples'][features]
    cross_validate_classifiers(classifiers, post_train, post_train_target, args.folds)
    
if __name__ == '__main__':
    main(sys.argv[1:])

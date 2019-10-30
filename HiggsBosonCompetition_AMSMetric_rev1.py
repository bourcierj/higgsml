# -*- coding: utf-8 -*-
"""
Evaluation metric for the Higgs Boson Kaggle Competition,
as described on:
https://www.kaggle.com/c/higgs-boson/details/evaluation

@author: Joyce Noah-Vanhoukce
Created: Thu Apr 24 2014
"""
import os
import csv
import math


def create_solution_dictionary(solution):
    """ Read solution file, return a dictionary with key EventId and value (weight,label).
    Solution file headers: EventId, Label, Weight """

    solnDict = {}
    with open(solution, 'r') as f:
        soln = csv.reader(f)
        next(soln) # header
        for row in soln:
            if row[0] not in solnDict:
                solnDict[row[0]] = (row[1], row[2])
    return solnDict


def check_submission(submission, Nelements):
    """ Check that submission RankOrder column is correct:
        1. All numbers are in [1,NTestSet]
        2. All numbers are unqiue
    """
    rankOrderSet = set()
    with open(submission, 'r') as f:
        sub = csv.reader(f)
        next(sub) # header
        for row in sub:
            rankOrderSet.add(row[1])

    if len(rankOrderSet) != Nelements:
        print('RankOrder column must contain unique values')
        exit()
    elif rankOrderSet.isdisjoint(set(range(1,Nelements+1))) == False:
        print('RankOrder column must contain all numbers from [1..NTestSset]')
        exit()
    else:
        return True


def AMS(s, b):
    """ Approximate Median Significance defined as:
        AMS = sqrt(
                2 { (s + b + b_r) log[1 + (s/(b+b_r))] - s}
              )
    where b_r = 10, b = background, s = signal, log is natural logarithm """

    br = 10.0
    radicand = 2 *( (s+b+br) * math.log (1.0 + s/(b+br)) -s)
    if radicand < 0:
        print('radicand is negative. Exiting')
        exit()
    else:
        return math.sqrt(radicand)


def AMS_metric(solution, submission):
    """  Prints the AMS metric value to screen.
    Solution File header: EventId, Class, Weight
    Submission File header: EventId, RankOrder, Class
    """

    numEvents = 550000 # number of events = size of test set

    # solutionDict: key=eventId, value=(label, class)
    solutionDict = create_solution_dictionary(solution)

    signal = 0.0
    background = 0.0
    if check_submission(submission, numEvents):
        with open(submission, 'r') as f:
            sub = csv.reader(f)
            next(sub) # header row
            for row in sub:
                if row[2] == 's': # only events predicted to be signal are scored
                    if solutionDict[row[0]][0] == 's':
                        signal += float(solutionDict[row[0]][1])
                    elif solutionDict[row[0]][0] == 'b':
                        background += float(solutionDict[row[0]][1])

        print('signal = {0}, background = {1}'.format(signal, background))
        print('AMS = ' + str(AMS(signal, background)))


import pandas as pd

def build_solution():
    """Builds the solution file (columns EventId, Label, Weight) from the test csv fil
       with all other columns"""

    """Reads CSV file and splits in into four datasets: Train, Kaggle public
    eaderboard, Kaggle private leaderboard (i.e the test set), and unused."""
    df = pd.read_csv("data/atlas-higgs-challenge-2014-v2.csv", sep=',')
    df_test = df[df['KaggleSet'].isin(('b', 'v'))]
    assert(len(df_test) == 550_000)
    df_test = df_test.drop('Weight', axis='columns')
    df_test = df_test.rename(columns={'KaggleWeight': 'Weight', 'Label': 'Class'})
    df_test.to_csv('data/solution_from_cern.csv', columns=['EventId', 'Class', 'Weight'], index=False)


if __name__ == "__main__":

    # Check if test set from CERN and from Kaggle match: => Looks like they match

    # df = pd.read_csv("atlas-higgs-challenge-2014-v2.csv", sep=',')
    # dft1 = df[df['KaggleSet'].isin(('b', 'v'))]
    # assert(len(dft1) == 550_000)
    # dft1.drop('Weight', axis='columns')
    # dft1.rename(columns={'KaggleWeight': 'Weight'})
    # print(dft1.head(5))

    # dft2 = pd.read_csv("test.csv", sep=',')
    # print(dft2.head(5))

    # builds the solution file
    # build_solution()

    # enter path and file names here
    # solution_path = ''
    # submission_path = ''

    # AMS_metric(solution_path, submission_path)

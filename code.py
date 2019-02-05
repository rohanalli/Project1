# Example of kNN implemented from Scratch in Python

import csv
import random
import math
import operator
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler as Scaler
import time

def loadDataset(filename, split, trainingSet=[] , testSet=[]):
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(1,len(dataset)-1):
            for y in range(9):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    # print(length)
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

def processData(filename, split, trainingSet, testSet):
    DATASET_PATH = './'
    data_path = os.path.join(DATASET_PATH, filename)
    dataset = pd.read_csv(data_path, header=None)
    dataset.columns = [
        "Pregnancies", "Glucose", "BloodPressure",
        "SkinThickness", "Insulin", "BMI",
        "DiabetesPedigreeFunction", "Age", "Outcome"]

    median_bmi = dataset['BMI'].median()
    dataset['BMI'] = dataset['BMI'].replace(
        to_replace=0, value=median_bmi)


    median_bloodp = dataset['BloodPressure'].median()
    dataset['BloodPressure'] = dataset['BloodPressure'].replace(
        to_replace=0, value=median_bloodp)

    median_plglcconc = dataset['Glucose'].median()
    dataset['Glucose'] = dataset['Glucose'].replace(
        to_replace=0, value=median_plglcconc)

    median_skinthick = dataset['SkinThickness'].median()
    dataset['SkinThickness'] = dataset['SkinThickness'].replace(
        to_replace=0, value=median_skinthick)

    median_twohourserins = dataset['Insulin'].median()
    dataset['Insulin'] = dataset['Insulin'].replace(
        to_replace=0, value=median_twohourserins)
    # print(dataset.head())
    # print(dataset['BMI'][0])
    scaler = Scaler()
    scaler.fit(dataset)
    scaled_dataset = scaler.transform(dataset)
    # print(dataset['BMI'][0])
    df = pd.DataFrame(data=scaled_dataset)
    # print(df.head())
    # datasetlst = pd.
    # datasetlst = list(dataset)
    datasetlst = df.values.tolist()
    # map(datasetlst,dataset.values)
    # datasetlst = list(dataset.values)
    # print(datasetlst[0][5])
    print(len(datasetlst))
    for x in range(0,len(datasetlst)-1):
        for y in range(9):
            datasetlst[x][y] = float(datasetlst[x][y])
        if random.random() < split:
            trainingSet.append(datasetlst[x])
        else:
            testSet.append(datasetlst[x])

def printTable(predictions):
    fp=0
    fn=0
    tp=0
    tn=0
    # print(predictions)
    # print(predictions[0][1])
    for x in range(0,len(predictions)-1):
        if(predictions[x][0] == predictions[x][1]):
            if(predictions[x][0] == 0):
                tn+=1
            else:
                tp+=1
        if(predictions[x][0] != predictions[x][1]):
            if(predictions[x][0] == 0):
                fn+=1
            else:
                fp+=1
    print(tp,tn,fp,fn)
    accuracy = float(float(tp+tn)/float(tp+tn+fp+fn)*100.0)
    print(accuracy)

def main():
    # prepare data
    trainingSet=[]
    testSet=[]
    split = 0.60
    start = time.time()
    processData('diabetes1.csv', split, trainingSet, testSet)
    print 'Train set: ' + repr(len(trainingSet))
    print 'Test set: ' + repr(len(testSet))
    # generate predictions
    accupredictions=[]
    predictions=[]
    k = 5
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        accupredictions.append(result)
        predictions.append([result,testSet[x][-1]])
        # print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    # print(predictions)
    accuracy = getAccuracy(testSet, accupredictions)
    end = time.time()
    print('Accuracy: ' + repr(accuracy) + '%')
    printTable(predictions)
    print(end-start)

main()

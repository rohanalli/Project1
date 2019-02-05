# Example of kNN implemented from Scratch in Python

import csv
import random
import math
import operator
import os
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
from progressbar import AnimatedMarker, Bar, BouncingBar, Counter, ETA, AdaptiveETA, FileTransferSpeed, FormatLabel, Percentage, ProgressBar, ReverseBar, RotatingMarker, SimpleProgress, Timer, UnknownLength
pbar = ProgressBar()
from sklearn.preprocessing import MinMaxScaler as Scaler
import time

# import hashmap

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((float(instance1[x]) - float(instance2[x])), 2)
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
        response = neighbors[x][0]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
    correct = 0
    # print(predictions)
    # print(len(testSet))
    for x in range(len(testSet)):
        if testSet[x][0] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

def processData(filename, split, trainingSet, validationSet,testSet, length):
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        print("Total set - "+str(len(dataset)))
        # print(repr(len(dataset[0])))
        bool = True
        for x in range(1,5000):
            for y in range(len(dataset[x])):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split[0]:
                trainingSet.append(dataset[x])
            elif bool == True:
                validationSet.append(dataset[x])
                bool = False
            elif bool == False:
                testSet.append(dataset[x])
                bool = True

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
    validationSet=[]
    testSet=[]
    split = [0.80,0.90]
    # start = time.time()
    processData('train.csv', split, trainingSet, validationSet,testSet,5000)
    print 'Train set: ' + repr(len(trainingSet))
    print 'validation set: ' + repr(len(validationSet))
    print 'test set: ' + repr(len(testSet))
    print("\n")
    # generate predictions

    k_accuracy=[]
    for k in range(1,23,2):
        accupredictions=[]
        predictions=[]
        widgets = [Percentage(),
                   ' ', Bar(),
                   ' ', ETA(),
                   ' ', AdaptiveETA()]
        pbar = ProgressBar(widgets=widgets, maxval=len(validationSet))
        pbar.start()
        # print("K value is - "+repr(k))
        for x in range(0,len(validationSet)):
            neighbors = getNeighbors(trainingSet, validationSet[x], k)
            result = getResponse(neighbors)
            accupredictions.append(result)
            predictions.append([result,validationSet[x][0]])
            pbar.update(x+1)
            # print('> predicted=' + repr(result) + ', actual=' + repr(validationSet[x][0]))
        # print("predictions for k value - "+repr(k)+" is "+repr(predictions))
        pbar.finish()
        accuracy = getAccuracy(validationSet, accupredictions)
        # end = time.time()
        k_accuracy.append(accuracy)
        print('Accuracy: for k - '+repr(k)+ " is " + repr(accuracy) + '%')
        print("\n")

    print("\n")
    # print(k_accuracy)
    k_val = k_accuracy.index(max(k_accuracy))+1
    print("Best Accuracy is for k = "+repr(k_val))
    print("\n")
    #
    #
    # # processData('test.csv',1,testSet,[],20)
    # print(k_val)
    accupredictions_t=[]
    predictions_t=[]
    widgets = [Percentage(),
               ' ', Bar(),
               ' ', ETA(),
               ' ', AdaptiveETA()]
    pbar = ProgressBar(widgets=widgets, maxval=len(testSet))
    pbar.start()
    for x in range(0,len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        # print(len(testSet[x]))
        result = getResponse(neighbors)
        accupredictions_t.append(result)
        predictions_t.append([result,testSet[x][0]])
        pbar.update(x+1)
        # print('> predicted=' + repr(result) + ', actual=' + repr(validationSet[x][0]))
    pbar.finish()
    # print("predictions for k value - "+repr(k_val)+" is "+repr(predictions_t))
    accuracy = getAccuracy(testSet, accupredictions_t)
    print("Final test Accuracy of KNN for k - value "+repr(k_val)+" is " + repr(accuracy) + '%')
    confusion_matrix = [[0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0]]

    for t in range(len(predictions_t)):
        i = int(predictions_t[t][0])
        j = int(predictions_t[t][1])
        confusion_matrix[i][j] += 1

    df_cm = pd.DataFrame(confusion_matrix, range(10),
                  range(10))
    #plt.figure(figsize = (10,7))
    sn.set(font_scale=1.4)#for label size
    sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size
    plt.show()

main()

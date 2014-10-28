'''
Created on Oct 15, 2014

@author: eotles
'''
import collections
import math
import random
import sys
attribute = collections.namedtuple('attribute', ['name', 'isNominal', 'nominalValues'])
arff = collections.namedtuple('arff', ['relation', 'attributes', 'data'])
performance = collections.namedtuple('performance', ['truePos',  'trueNeg','falsePos', 'falseNeg'])

class perceptron(object):
    def __init__(self, attributes):
        self.attributes = attributes
        self.features =  [x.name for x in attributes[:-1]]
        self.weights = [0.1 for _ in self.features]
        self.bias = 0.1
    
    #return predicted output based on instance
    #NOTE: output label is numerical
    #NOTE: instance is whole vector of instance data (including label, which isn't used)
    def prediction(self, instance):
        sumprod = self.bias
        for index, weight in enumerate(self.weights):
            sumprod += weight*instance[index]
        output = 1/(1+math.e**-sumprod)
        return(output)

    #update perceptron based on new instance
    #NOTE: label is numerical
    #NOTE: instance is whole vector of instance data (including label, which isn't used)    
    def update(self, instance, label, lr):
        output = self.prediction(instance)
        delta = -(label-output)*output*(1-output)
        self.bias -= lr*delta*self.bias
        for index, weight in enumerate(self.weights):
            self.weights[index] -= lr*delta*instance[index]

    #train perceptron
    #NOTE: since perceptron needs label to be numerical turning nominal values into 0 or 1
    def train(self, data, lr,  epochs):
        for _ in range(epochs):
            for instance in data:
                label = 0 if(instance[-1] == self.attributes[-1].nominalValues[0]) else 1
                self.update(instance, label, lr)
    
    #evaluate performance of perceptron on testing data 
    #NOTE: since perceptron returns numerical prediction need to conversion into nominal label is done    
    def evaluatePerformance(self, data):
        tp = fp = tn = fn = 0
        for instance in data:
            predictedLabel = self.prediction(instance)
            predictedLabel = self.attributes[-1].nominalValues[0] if(predictedLabel <= 0.5) else self.attributes[-1].nominalValues[1]
            actualLabel = instance[-1]
            if(predictedLabel == actualLabel):
                if(actualLabel == self.attributes[-1].nominalValues[1]): tp += 1
                else: tn += 1
            else:
                if(predictedLabel == self.attributes[-1].nominalValues[1]): fp += 1
                else: fn += 1
        return(performance(tp, tn, fp, fn))
                
                
#actually train & eval perceptron based on k-folds
def trainAndTestPerceptron(arff, folds, lr, epochs, compAcc, dispLines):
    trainPer = list()
    testPer = list()
    
    #setup and train perceptrons for each fold
    perceptrons = [perceptron(arff.attributes) for _ in folds]
    for foldID, fold in enumerate(folds):
        trainIDs = [x for potentialFold in folds[:foldID]+folds[foldID+1:] for x in potentialFold]
        testIDs = fold
        trainData = [arff.data[x] for x in trainIDs]
        testData = [arff.data[x] for x in testIDs]
        perceptrons[foldID].train(trainData, lr, epochs)
        #for each fold evaluate performance
        trainPer.append(perceptrons[foldID].evaluatePerformance(trainData))
        testPer.append(perceptrons[foldID].evaluatePerformance(testData))
    
    #print average accuracy if requested
    if(compAcc):
        print("\t%s, %s" %(totalAcc(trainPer), totalAcc(testPer)))
    
    #print all data with fold, predicted label, actual label, and output if requested    
    if(dispLines):
        for index, instance in enumerate(arff.data):
            acutalLabel = instance[-1]
            for foldID, fold in enumerate(folds):
                if(index in fold):
                    output = perceptrons[foldID].prediction(instance)
                    predictedLabel = arff.attributes[-1].nominalValues[0] if(output <= 0.5) else arff.attributes[-1].nominalValues[1]
                    print("%d\t%s\t%s\t%f" %(foldID+1, predictedLabel, acutalLabel, output))
                    break

#compute accuracy and accuracy sd based on folds performance
def totalAcc(foldsPerformance):
    foldsAcc = [float(performance.truePos + performance.trueNeg) / 
                (performance.truePos + performance.trueNeg + performance.falsePos + performance.falseNeg)
                for performance in foldsPerformance] 
    mean = sum(foldsAcc)/len(foldsAcc)
    sd = 0
    for val in foldsAcc:
        sd += (val - mean)**2
    sd = math.sqrt(sd/len(foldsAcc))
    return("%f %f"  %(mean, sd))  

#create folds based on stratified cross validation                
def strataCrossValSelection(k, arff):
    strata = dict()
    strataCount = dict()
    partitions = [list() for _ in range(k)]
    
    #build a dictionary with each label value pointing to a list of data row 
    #indices that have classification label
    for nomVal in arff.attributes[-1].nominalValues:
        strata.update({nomVal: []})
    for index,line in enumerate(arff.data):
        label = line[-1]
        strataList = strata.get(label)
        strataList.append(index)
        strata.update({label: strataList})
    
    #counts of each label - needed for stratification         
    for label, strataList in strata.iteritems():
        strataCount.update({label: len(strataList)})
    
    #randomly* put indexes in equal sized partitions
    #*stratification must ensure they have the same class distribution breakdown 
    for part in partitions:
        for label, strataList in strata.iteritems():
            numNeeded = strataCount.get(label)/k
            for _ in range(numNeeded):
                choosenDataIndex = random.choice(strataList)
                strataList.remove(choosenDataIndex)
                part.append(choosenDataIndex)
            strata.update({label: strataList})
    
    #there may be leftover row data indices, distribute those
    leftOvers = list()
    for label, strataList in strata.iteritems():
        leftOvers += strataList
    for part in partitions:
        if(len(leftOvers) <= 0):
            break
        else:
            choosenDataIndex = random.choice(leftOvers)
            leftOvers.remove(choosenDataIndex)
            part.append(choosenDataIndex)
            
    return(partitions)

#code for doing hw questions
def doProblems():
    n = 10
    l = 0.1
    e = 100
    random.seed(42)
    tf = "/Users/eotles/Documents/workspace/CS760/hw2/sonar.arff"
    learnARFF = load(tf)
    print(learnARFF.attributes)
    folds = strataCrossValSelection(n, learnARFF)
    print("Acc")
    for e in [1, 10, 100, 1000]:
        print("\tEpochs: %s" %(e))
        trainAndTestPerceptron(learnARFF, folds, l, e, True, False)
    print("\nROC")
    trainAndTestPerceptron(learnARFF, folds, l, 100, False, True)

#load file into custom ARFF framework
def load(filePath):    
    trainDataFile = open(filePath)
    headerMode = True
    relation = ""
    attributes = list()
    data = list()
    for line in trainDataFile:
        line = line.strip()
        if(headerMode):
            tag = line[:10].upper()
            if tag.startswith("@RELATION"):
                relation = line[10:]
            elif tag.startswith("@ATTRIBUTE"):
                attributeName = line.split(" ")[1].strip("'")
                attributeIsNominal = False
                attributeNominalValues = None
                openBracketPos = line.find("{")
                if(openBracketPos > 0):
                    attributeIsNominal = True
                    closeBracketPos = line.find("}")
                    attributeNominalValues = line[openBracketPos+1:closeBracketPos].split(",")
                    attributeNominalValues = [anv.strip(" ") for anv in attributeNominalValues]
                attributes.append(attribute(attributeName, attributeIsNominal, attributeNominalValues))
            elif tag.startswith("@DATA"):
                headerMode = False
        else:
            line = line.split(",")
            line = [val if attributes[index].isNominal else float(val) for index,val in enumerate(line)]
            data.append(line)       
    return(arff(relation, attributes, data))

def main(dataSetFilePath, n, l, e):
    #doProblems()
    dataSet = load(dataSetFilePath)
    folds = strataCrossValSelection(n, dataSet)
    trainAndTestPerceptron(dataSet, folds, l, e, compAcc=False, dispLines=True)
    

if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]))
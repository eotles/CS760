'''
Created on Sep 18, 2014

@author: eotles
'''

import collections
import copy
import math
import random
import sys
import attributes
from _random import Random
attribute = collections.namedtuple('attribute', ['name', 'isNominal', 'nominalValues'])
arff = collections.namedtuple('arff', ['relation', 'attributes', 'data'])
attClassPair = collections.namedtuple('attClassPair', ['attVal', 'classVal'])
nonNominalSplit = collections.namedtuple('nonNominalSplit', ['conditionalEntropy', 'midpoint'])
split = collections.namedtuple('split', ['attributeIndex', 'midpoint'])
voteResults = collections.namedtuple('voteResults', ['tally', 'winner'])
LOG2 = math.log(2)

ATTRIBUTES = None

class dtNode(object):
    def __init__(self):
        self.nodeInfo = None
        self.isLeaf = None
        self.predClass = None
        self.tally = None
        self.split = None
        self.children = dict()
    
    def makeLeaf(self, predClass):
        self.isLeaf = True
        self.predClass = predClass
    
    def makeInterior(self, split):
        self.isLeaf = False
        self.split = split
    
    def addChild(self, attVal, childNode):
        #print "add child:",
        #print attVal,
        #print childNode
        self.children.update({attVal : childNode})
    
    def toDisp(self, attributes, path, level):
        outString = ""
        for _ in xrange(level):
            outString += "|\t"
        outString += path
        outString += " " + str(self.nodeInfo)
        if(self.isLeaf):
            outString += ": " + str(self.predClass)
            print(outString)
            return()
        else:
            print(outString)
            rootSplitAtt = attributes[self.split.attributeIndex]
            if(rootSplitAtt.isNominal):
                for val in rootSplitAtt.nominalValues:
                    path = rootSplitAtt.name + " = " + val
                    child = self.children.get(val)
                    child.toDisp(attributes, path, level+1)
            else:
                mid = self.split.midpoint
                for val in [True, False]:
                    sign = " <= " if val else " >  " 
                    path = rootSplitAtt.name + sign + str(mid)
                    child = self.children.get(val)
                    child.toDisp(attributes, path, level+1)

def percentData(percent, data):
    numNeeded = min(percent,1)*len(data)
    availIndices = range(len(data))
    pickedData = list()
    while(len(pickedData) < numNeeded):
        getIndex = random.choice(availIndices)
        availIndices.remove(getIndex)
        pickedData.append(data[getIndex])
    return(pickedData)
    


def test(testFilePath, dtRootNode):
    testARFF = load(testFilePath)
    numCor = 0
    numIncor = 0
    for instance in testARFF.data:
        print evaluate(dtRootNode, testARFF.attributes, instance),
        print " ",
        print(instance[-1])
        if(evaluate(dtRootNode, testARFF.attributes, instance) == instance[-1]):
            numCor+=1
        else:
            numIncor+=1
    print(str(numCor) + " " + str(numCor+numIncor))
    accuracy = float(numCor)/(numCor+numIncor)
    print("Accuracy: " + str(accuracy))
    return(accuracy)
            
    
def evaluate(dtRootNode, attributes, instance):
    currDTNode = dtRootNode
    if(currDTNode.isLeaf):
        return(currDTNode.predClass)
    else:
        attSplit = currDTNode.split
        attIndex = attSplit.attributeIndex
        attVal = instance[attIndex]
        if(attributes[attIndex].isNominal):
            currDTNode = currDTNode.children.get(attVal)
        else:
            currDTNode = currDTNode.children.get(attVal <= attSplit.midpoint)
        return(evaluate(currDTNode, attributes, instance))

def learn(learnARFF, m):
    ATTRIBUTES = learnARFF.attributes
    transData = transpose(learnARFF.attributes, learnARFF.data)
    root = buildTree(transData, learnARFF.attributes, m)
    rootSplitAtt = ATTRIBUTES[root.split.attributeIndex]
    if(rootSplitAtt.isNominal):
        for val in rootSplitAtt.nominalValues:
            path = rootSplitAtt.name + " = " + val
            child = root.children.get(val)
            child.toDisp(ATTRIBUTES, path, 0)
    else:
        mid = root.split.midpoint
        for val in [True, False]:
            sign = " <= " if val else " < " 
            path = rootSplitAtt.name + sign + str(mid)
            child = root.children.get(val)
            child.toDisp(ATTRIBUTES, path, 0)
    return(root)

def buildTree(transData, attributes, m):
    def makePred(classCol, attributes):
        predCount = dict()
        for cla in attributes[-1].nominalValues:
            predCount.update({cla : 0})
        for cla in classCol:
            predCount.update({cla : predCount.get(cla)+1})
        tally = []
        maxVotes = -1
        pred = None
        for cla in attributes[-1].nominalValues:
            votes = predCount.get(cla)
            tally.append(votes)
            if(votes > maxVotes):
                maxVotes = votes
                pred = cla
        results= voteResults(tally, pred)
        return(results)
            
    #print("new node")
    currNode = dtNode()
    pred = makePred(transData[-1], attributes)
    currNode.tally = pred.tally
    currNode.nodeInfo = pred.tally
    
    #stopping criteria
    #print(transData[-1])
    if(len(set(transData[-1])) == 1):
        pred = makePred(transData[-1], attributes)
        currNode.tally = pred.tally
        currNode.makeLeaf(pred.winner)
        return(currNode)
    elif(len(transData[-1]) < m):
        pred = makePred(transData[-1], attributes)
        currNode.tally = pred.tally
        currNode.makeLeaf(pred.winner)
        return(currNode)
    elif(len(attributes) <= 0):
        pred = makePred(transData[-1], attributes)
        currNode.tally = pred.tally
        currNode.makeLeaf(pred.winner)
        return(currNode)
    else:
        bestSplit = findBestSplit(transData, attributes)
        currNode.makeInterior(bestSplit)
        #print "\tBestSplit@",
        #print attributes[bestSplit.attributeIndex],
        #print bestSplit.attributeIndex,
        #print("| midpoint:" + str(bestSplit.midpoint))
        selectedAttributeIndex = bestSplit.attributeIndex
        splitAttributes = copy.copy(attributes)
        if(splitAttributes[bestSplit.attributeIndex].isNominal):
            splitData = splitDataNomninally(transData, splitAttributes, selectedAttributeIndex)
            #splitAttributes.pop(selectedAttributeIndex)
        else:
            midpoint = bestSplit.midpoint
            splitData = splitDataByPoint(transData, splitAttributes, selectedAttributeIndex, midpoint)
        
        for val,subNodeData in splitData.iteritems():
            #print "\n",
            #print(val)
            currNode.addChild(val, buildTree(subNodeData, splitAttributes, m)) 
        return(currNode)
    
def splitDataNomninally(data, attributes, selectedAttributeIndex):
    splitData = dict()
    #carvedData = data[:selectedAttributeIndex] + data[selectedAttributeIndex+1:]
    carvedData = data
    for nominalVal in attributes[selectedAttributeIndex].nominalValues:
        splitData.update({nominalVal : [ [] for _ in carvedData ]})
    for colIndex, col in enumerate(carvedData):
        for instanceIndex, instanceVal in enumerate(data[selectedAttributeIndex]):
            storedData = splitData.get(instanceVal)
            storedData[colIndex].append(col[instanceIndex])
            splitData.update({instanceVal : storedData})  
    return(splitData)      

def splitDataByPoint(data, attributes, selectedAttributeIndex, midpoint):
    splitData = dict()
    for ltMidpoint in [True, False]:
        splitData.update({ltMidpoint : [ [] for _ in data ]})
    ltMidpointList = []
    for attVal in data[selectedAttributeIndex]:
        ltMidpointList.append(attVal <= midpoint) 

    for colIndex,col in enumerate(data):
        for instanceIndex, ltMidpointVal in enumerate(ltMidpointList):
            storedData = splitData.get(ltMidpointVal)
            storedData[colIndex].append(data[colIndex][instanceIndex])
            splitData.update({ltMidpointVal : storedData})
    return(splitData)

   
def findBestSplit(transData, attributes):
    minCondEntropyVal = sys.float_info.max
    minCondEntropyAtt = None    
    for index,attCol in enumerate(transData[:-1]):
        if(attributes[index].isNominal):
            condEntropy = computeCondEntropy(attCol, transData[-1])
        else:
            nns = findNonNomialSplit(attCol, transData[-1])
            condEntropy = nns.conditionalEntropy
            midpoint = nns.midpoint
        #print(attributes[index].name),
        #print ": ",
        #print condEntropy
        if(condEntropy < minCondEntropyVal):
            minCondEntropyAttIndex = index
            minCondEntropyVal = condEntropy
            minCondEntropyAtt = attributes[index]
            if(attributes[index].isNominal):
                minCondEntropyAttMidpoint = None
            else:
                minCondEntropyAttMidpoint = midpoint
    return(split(minCondEntropyAttIndex, minCondEntropyAttMidpoint))


def findNonNomialSplit(attCol, classCol):
    def potentialSplitConditionalEntropy(attCol, classCol, midpoint):
        bAttCol = [(val <= midpoint) for val in attCol]
        return(computeCondEntropy(bAttCol, classCol))
    
    numberLine = dict()
    for index,attVal in enumerate(attCol):
        labels = set()
        if(numberLine.has_key(attVal)):
            labels = numberLine.get(attVal)
        labels.add(classCol[index])
        numberLine.update({attVal: labels})
        
    numberLineVals = sorted(numberLine)
    bestCondEntropy = sys.float_info.max
    bestSplit = None
    lastVal = numberLineVals[0]
    lastValLabel = numberLine.get(lastVal)
    if(len(numberLineVals)<=1):
        midpoint = lastVal
        potentialEntropy = potentialSplitConditionalEntropy(attCol, classCol, midpoint)
        bestSplit = nonNominalSplit(potentialEntropy, midpoint) 
    else:
        for val in numberLineVals[1:]:
            valLabel = numberLine.get(val)
            if((valLabel != lastValLabel) or (len(valLabel)>1 and len(lastValLabel)>1)):
                midpoint = (val + lastVal)/2
                potentialEntropy = potentialSplitConditionalEntropy(attCol, classCol, midpoint)
                if(potentialEntropy < bestCondEntropy):
                    bestCondEntropy = potentialEntropy
                    bestSplit = nonNominalSplit(potentialEntropy, midpoint)      

            lastVal = val
            lastValLabel = valLabel
    return(bestSplit)
       

def computeCondEntropy(attCol, classCol):
    aDict = dict()
    nested = dict()
    condEntropy = 0
    for index,attVal in enumerate(attCol):
        classVal = classCol[index]
        aCount = 1
        if(aDict.has_key(attVal)):
            aCount = aDict.get(attVal)+1
        aDict.update({attVal : aCount})
        nCount = 1
        if(nested.has_key(attVal)):
            if(nested.get(attVal).has_key(classVal)):
                nCount = nested.get(attVal).get(classVal)+1
        else:
            nested.update({attVal : dict()})
        nested.get(attVal).update({classVal : nCount})

    for x,c_x in aDict.iteritems():
        p_x = float(c_x)/len(attCol)
        y_entropy = 0
        for y,c_y in nested.get(x).iteritems():
            p_y1x = float(c_y)/c_x
            y_entropy -= p_y1x * math.log(p_y1x)/LOG2
        condEntropy += p_x*y_entropy
    return(condEntropy)
            
    
def transpose(attributes, data):
    transData = [list() for _ in attributes ]
    for row in data:
        for col, value in enumerate(row):
            transData[col].append(value)
    return(transData)
    
def load(filePath):    
    trainDataFile = open(filePath)
    headerMode = True
    relation = ""
    attributes = list()
    data = list()
    id = 0
    for line in trainDataFile:
        line = line.strip()
        if(headerMode):
            tag = line[:10].upper()
            #print(tag)
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
            id+=1         
    return(arff(relation, attributes, data))
        
def main(trainFilePath, testFilePath, m):
    learnARFF = load(trainFilePath)
    dtRootNode = learn(learnARFF, m)
    test(testFilePath, dtRootNode)
    
    '''
    learnData = copy.copy(learnARFF.data)
    accuracies = list()
    accDict = dict()
    random.seed(42)
    for perc in [1]:
        for m in [2,5,10, 20]:
            accuracies = list()
            for _ in xrange(10):
                expARFF = arff(learnARFF.relation, learnARFF.attributes, percentData(perc, learnData))
                expDTRootNode = learn(expARFF, m)
                accuracies.append(test(testFilePath, expDTRootNode))
            accDict.update({str(perc)+"|"+str(m): accuracies})
    for key, val in accDict.iteritems():
        print key,
        print "\t",
        print str(val).strip("[").strip("]")
    '''

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
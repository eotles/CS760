'''
Created on Sep 19, 2014

@author: eotles
'''
import abc

class attributes(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.data = list()
        
    def setData(self, data):
        self.data = data
        
    def apend(self, datum):
        self.data.append(datum)
        
class numeric_attributes(attributes):
    def __init__(self, data):
        super(numeric_attributes, self).__init__(data)
    
    def apend(self, datum):
        super(numeric_attributes, self).append(datum)


    
    

    


        
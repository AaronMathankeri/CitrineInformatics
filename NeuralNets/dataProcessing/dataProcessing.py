from sys import argv
script, filename = argv
# ------------------------------------------------------------
# read unique compounds create a hash

#print "Total Number of Different elements %r:" % filename
print "Creating Hash Table ..."
elements = [line.rstrip('\n') for line in open(filename)]
#for x in elements:
#    print x

#number codes for elements
Dictionary = {}

#print len(elements)
for idx in range( len(elements) ):
    Dictionary[elements[idx]] = (idx + 1) 

#for k in Dictionary.keys():
#    print Dictionary[k]

#for k,v in Dictionary.items():
#    print k, ':',v
print "Complete."

# ------------------------------------------------------------
#now read csv file and transform elements with new dictionary!
print "Importing and transforming data..."
import pandas as pd
import numpy as np

df = pd.read_csv('test_data.csv')

#print df

formulaA = df['formulaA']
formulaB = df['formulaB']
compoundsA = []
compoundsB = []

for idx in range(len(formulaA)):
    compoundsA.append( Dictionary[formulaA[idx]])
    compoundsB.append( Dictionary[formulaB[idx]])

df['formulaA'] = compoundsA
df['formulaB'] = compoundsB

#print df['formulaA']
#print df['formulaB']

#targetVec = df['stabilityVec']
#print targetVec
print "Complete."
# ------------------------------------------------------------
# save transformed data to file for Torch
print "Writing to File..."

#remove last column because I/O in torch is a Bitch
#targetVec = df['stabilityVec']
#targetVec.to_csv("target_training_data.csv", index=False)

#df = df.drop(labels='stabilityVec', axis=1)
df.to_csv("feature_test_data.csv", index=False)



print "Complete."


#-------------------------------------------------------------------------
# AUTHOR: Jessica Ortega
# FILENAME: naive_bayes.py
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB

dbTraining = []
X = []
Y = []
#reading the training data in a csv file
#--> add your Python code here
with open('weather_training.csv', 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)

#transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
# X =

outlook = {
    "Sunny" : 1,
    "Overcast" : 2,
    "Rain" : 3, 
}

temperature = {
    "Hot" : 1,
    "Mild" : 2,
    "Cool" : 3, 
}

humidity = {
    "High" : 1,
    "Low" : 2, 
}
wind = {
    "Strong" : 1,
    "Weak" : 2,
}
for row in dbTraining:
        X.append([outlook[row[0]], temperature[row[1]], humidity[row[2]], wind[row[3]]])

#transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
# Y =
playTennis = {
    "Yes" : 1,
    "No" : 2, 
}

for row in dbTraining:
        Y.append([playTennis[row[4]]])

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the test data in a csv file
#--> add your Python code here
dbTest = []
with open('weather_test.csv', 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)

#printing the header os the solution
#--> add your Python code here

#use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here



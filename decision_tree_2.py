#-------------------------------------------------------------------------
# AUTHOR: Jessica Ortega 
# FILENAME: decision_tree_2.py
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []
    accuracy = []

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)

    #transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
    # so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
    # X =

    age = {
        "Young" : 1,
        "Prepresbyopic" : 2,
        "Presbyopic" : 3,
    }
    spectaclePrescription = {
        "Myope" : 1,
        "Hypermetrope" : 2,
    }
    astigmatism = {
        "Yes" : 1,
        "No" : 2,
    }
    tearProductionRate = {
        "Reduced" : 1, 
        "Normal" : 2,
    }

    for row in dbTraining:
        X.append([age[row[0]], spectaclePrescription[row[1]], astigmatism[row[2]], tearProductionRate[row[3]]])

    #transform the original categorical training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    # Y =
    recommendedLenses = {
        "Yes" : 1,
        "No" : 2,
    }
    
    for row in dbTraining:
        Y.append([recommendedLenses[row[4]]])

    #loop your training and test tasks 10 times here

    TP = 0
    TN = 0
    FP = 0
    FN = 0


    for i in range (10):

       #fitting the decision tree to the data setting max_depth=3
       clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
       clf = clf.fit(X, Y)

       #read the test data and add this data to dbTest
       #--> add your Python code here
       # dbTest =

       dbTest = []
       with open("contact_lens_test.csv", 'r') as csvfile:
        reader = csv.reader(csvfile)
        for j, row in enumerate(reader):
            if j > 0: #skipping the header
                dbTest.append (row)


        for data in dbTest:

            #transform the features of the test instances to numbers following the same strategy done during training,
            #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
            #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            #--> add your Python code here

            instance = [age[data[0]], spectaclePrescription[data[1]], astigmatism[data[2]], tearProductionRate[data[3]]]
            prediction = clf.predict([instance])[0]
            trueValue = recommendedLenses[data[4]]

            #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            #--> add your Python code here

            if(prediction == trueValue and trueValue == 1):
                TP += 1
            elif(prediction == trueValue and trueValue == 2):
                TN += 1
            if(prediction != trueValue and trueValue == 1):
                FP += 1
            elif(prediction != trueValue and trueValue == 2):
                FN += 1
                
    #find the average of this model during the 10 runs (training and test set)
    #--> add your Python code here
    
    accuracy.append((TP + TN) / (TP + TN + FP + FN))

    #print the average accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here
    print("Final accuracy when training on " + ds + ": " + str(accuracy))




#-------------------------------------------------------------------------
# AUTHOR: Jessica Ortega
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #5
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('training_data.csv', sep=',', header=None) #reading the data by using Pandas library

#assign your training data to X_training feature matrix
X_training = df.values

#run kmeans testing different k values from 2 until 20 clusters
     #Use:  kmeans = KMeans(n_clusters=k, random_state=0)
     #      kmeans.fit(X_training)
     #--> add your Python code

silhouette_scores = []
best_silhouette_score = -1
best_k = 0

for k in range(2, 21):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_training)

     #for each k, calculate the silhouette_coefficient by using: silhouette_score(X_training, kmeans.labels_)
     #find which k maximizes the silhouette_coefficient
     #--> add your Python code here
    silhouette_coefficient = silhouette_score(X_training, kmeans.labels_)
    silhouette_scores.append(silhouette_coefficient)

    if silhouette_coefficient > best_silhouette_score:
        best_silhouette_score = silhouette_coefficient
        best_k = k

#plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
#--> add your Python code here

plt.plot(range(2, 21), silhouette_scores)
plt.xlabel('K Values')
plt.ylabel('Silhouette Coefficient')
plt.title('Silhouette Coefficient')
plt.show()

#reading the test data (clusters) by using Pandas library
#--> add your Python code here
df_test = pd.read_csv('testing_data.csv', sep=',', header=None)

#assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]
#--> add your Python code here

labels = np.array(df_test.values).reshape(1, -1)[0]

#Calculate and print the Homogeneity of this kmeans clustering
kmeans = KMeans(n_clusters = best_k, random_state=0)
kmeans.fit(X_training)
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

with open('data/train_labels.csv') as labelsFile:
	trainLabels = pd.read_csv(labelsFile)

with open('data/train_Comb.csv') as featuresFile:
	trainComb = pd.read_csv(featuresFile)

"""# **Data** preparation"""

mydata = trainComb.drop('Id', axis=1)  # Separating out the data
labels = trainLabels["Class"].values.tolist()
features = []
for ind, data in mydata.iterrows():
	features.append(data.values.tolist())

train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.2, random_state=None)

scaler = StandardScaler()
# Fit on training set only.
scaler.fit(train_x)
# Apply transform to both the training set and the test set.
train_x = scaler.transform(train_x)
test_x = scaler.transform(test_x)
# Make an instance of the Model
pca = PCA(.99)
pca.fit(train_x)
print("Number of minimum number of principal components such that 99% of the variance is retained is", format(pca.n_components_))
# Apply the mapping (transform) 
train_x = pca.transform(train_x)
test_x = pca.transform(test_x)

"""# **SVM classifiers**"""

# Initializing Support vector machines (SVM) classifiers
clfSvmLinear = SVC(kernel='linear')
clfSvmPoly = SVC(kernel='poly')
clfSvmSigmoid = SVC(kernel='sigmoid')
clfSvmRbf = SVC(kernel='rbf')
# Training Support vector machines (SVM) classifiers
clfSvmLinear.fit(train_x, train_y)
clfSvmPoly.fit(train_x, train_y)
clfSvmSigmoid.fit(train_x, train_y)
clfSvmRbf.fit(train_x, train_y)
# Calculate scoring Support vector machines (SVM) classifiers
scoreSvmLinear = clfSvmLinear.score(test_x, test_y) * 100
scoreSvmPoly = clfSvmPoly.score(test_x, test_y) * 100
scoreSvmSigmoid = clfSvmSigmoid.score(test_x, test_y) * 100
scoreSvmRbf = clfSvmRbf.score(test_x, test_y) * 100
# Cross-validation
scoreSvmLinearCV = cross_val_score(clfSvmLinear, features, labels, cv=10).mean() * 100
scoreSvmPolyCV = cross_val_score(clfSvmPoly, features, labels, cv=10).mean() * 100
scoreSvmSigmoidCV = cross_val_score(clfSvmSigmoid, features, labels, cv=10).mean() * 100
scoreSvmRbfCV = cross_val_score(clfSvmRbf, features, labels, cv=10).mean() * 100
# Outputs Support vector machines (SVM) classifiers
print("Linear SVM Accuracy {}".format(scoreSvmLinear))
print("Mean score of cross validation in linear SVM {}".format(scoreSvmLinearCV))
print("Polynomial SVM Accuracy {}".format(scoreSvmPoly))
print("Mean score of cross validation in polynomial SVM {}".format(scoreSvmPolyCV))
print("Sigmoid SVM Accuracy {}".format(scoreSvmSigmoid))
print("Mean score of cross validation in sigmoid SVM {}".format(scoreSvmSigmoidCV))
print("Rbf SVM Accuracy {}".format(scoreSvmRbf))
print("Mean score of cross validation in rbf SVM {}".format(scoreSvmRbfCV))

"""# Decision Tree **Classifier**"""

from sklearn.tree import DecisionTreeClassifier

# Initializing Random Forest classifier
clfDT = DecisionTreeClassifier()
# Cross-validation
scoreDTCV = cross_val_score(clfDT, features, labels, cv=10).mean() * 100
# Training classifiers
clfDT.fit(train_x, train_y)
# Calculate scoring classifiers
scoreDT = clfDT.score(test_x, test_y) * 100
# Outputs
print("Decision Tree Accuracy {}".format(scoreDT))
print("Mean score of cross validation in decision tree {}".format(scoreDTCV))

"""# Random Forest **classifier**"""

# Initializing Random Forest classifier
clfRF = RandomForestClassifier()
# Cross-validation
scoreRFCV = cross_val_score(clfRF, features, labels, cv=10).mean() * 100
# Training classifiers
clfRF.fit(train_x, train_y)
# Calculate scoring classifiers
scoreRF = clfRF.score(test_x, test_y) * 100
# Outputs
print("Random Forest Accuracy {}".format(scoreRF))
print("Mean score of cross validation in random forest {}".format(scoreRFCV))

"""# K-Nearest Neighbours classifier (**KNN**)"""

# Initializing K-Nearest Neighbours classifier
clfKNN = KNeighborsClassifier(n_neighbors=3)
# Cross-validation
scoreKNNCV = cross_val_score(clfKNN, features, labels, cv=10).mean() * 100
# Training classifiers
clfKNN.fit(train_x, train_y)
# Calculate scoring classifiers
scoreKNN = clfKNN.score(test_x, test_y) * 100
# Outputs
print("K-Nearest Neighbours Accuracy {}".format(scoreKNN))
print("Mean score of cross validation in K-Nearest Neighbours {}".format(scoreKNNCV))

"""# Gaussian Naive **Bayes**"""

# Initializing K-Nearest Neighbours classifier
clfGNB = GaussianNB()
# Cross-validation
scoreGNBCV = cross_val_score(clfGNB, features, labels, cv=10).mean() * 100
# Training classifiers
clfKNN.fit(train_x, train_y)
# Calculate scoring classifiers
scoreKNN = clfKNN.score(test_x, test_y) * 100
# Outputs
print("K-Nearest Neighbours Accuracy {}".format(scoreKNN))
print("Mean score of cross validation in K-Nearest Neighbours {}".format(scoreKNNCV))

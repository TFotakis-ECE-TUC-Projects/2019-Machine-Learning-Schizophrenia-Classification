import matplotlib.pyplot as plt
import pandas as pd


def plotting():
	with open('data/train_labels.csv') as labelsFile:
		trainLabels = pd.read_csv(labelsFile)

	with open('data/train_Comb.csv') as featuresFile:
		trainComb = pd.read_csv(featuresFile)

	with open('data/train_SBM.csv') as featuresFile:
		trainSBM = pd.read_csv(featuresFile)

	"""# **Data** preparation"""

	mydata = trainComb.drop('Id', axis=1)  # Separating out the data
	labels = trainLabels["Class"].values.tolist()
	features = []
	for ind, data in mydata.iterrows():
		features.append(data.values.tolist())

	for j in [0, 4, 6, 7, 8]:
		fncPerId = [trainComb['FNC' + str(i)][j] for i in range(1, 378)]
		plt.plot(fncPerId)
		plt.ylabel('some numbers')
	plt.show()
	trainSBM = trainSBM.drop('Id', axis=1)
	for j in [0, 4, 6, 7, 8]:
		sbmPerId = [trainSBM[i][j] for i in list(trainSBM.keys())]
		plt.plot(sbmPerId)
		plt.ylabel('some numbers')
	plt.show()


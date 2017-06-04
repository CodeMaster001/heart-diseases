
import numpy
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn import svm
import sys
from sklearn.preprocessing import LabelEncoder


def preprocess_data(dataset):

	#remove nan values
	where_are_NaNs = np.isnan(dataset)
	dataset[where_are_NaNs] = 0;

	#shuffle numpy array
	np.random.shuffle(dataset)
	#remove lables and prepar train and test dataset
	X_train = np.delete(dataset, 1, -1)
	X_label_train = dataset[:,-1];
	print X_train.shape;
	print X_label_train.shape;
	return X_train,X_label_train;


def filter_non_disease(x):
    return  x[-1] > 0;

def process_labels(x):
	encoder = LabelEncoder()
	encoder.fit(x)
	Y = encoder.transform(x)
	return Y;


dataset= np.genfromtxt("dataset.csv",delimiter=",")
X,Y= preprocess_data(dataset);


bool_arr = np.array([filter_non_disease(row) for row in dataset])
dataset_disease= dataset[bool_arr]
X_D,Y_D=preprocess_data(dataset_disease);

clf = LogisticRegression(solver='lbfgs',class_weight='balanced',multi_class='ovr');

scores = cross_val_score(clf, X,Y, cv=10)
print np.mean(scores);

scores_D = cross_val_score(clf, X_D,Y_D, cv=10)
print np.mean(scores_D);

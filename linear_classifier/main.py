import pandas
import numpy
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

training_data = pandas.read_csv("../training_data/x_train_gr_smpl.csv")
labels = pandas.read_csv("../training_data/y_train_smpl.csv")
labels = numpy.ravel(labels)

# Always scale the input. The most convenient way is to use a pipeline.
linear_classifier = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
linear_classifier.fit(training_data, labels)
label_predictions = cross_val_predict(linear_classifier, training_data, labels, cv=10)
confusion_matrix = confusion_matrix(labels, label_predictions)
print(confusion_matrix)

run the predict.py with the out and test environment paramaters (test first)
(predict.py) fits the model from the test data and dumps the model in a text file using pickle

if you just wanna use the loaded model you can use loaded.py (same environment paramaters) 
which loads the model from lpq_classifier.txt and reads from the test_dir and writes the results in the 
out directory

Libraries Used:

cv2
numpy
glob
matpotlib
mpl_toolkits
sklearn.model_selection
skimage
scipy.optimize
scipy.signal import convolve2d
sklearn.svm import SVC
sklearn.metrics import classification_report,accuracy_score, confusion_matrix
sklearn.multiclass import OneVsRestClassifier
sklearn import metrics
sklearn.neighbors import KNeighborsClassifier
xgboost import XGBClassifier
time
sys
pickle
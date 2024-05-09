from joblib import dump
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = pd.read_csv("C:\\Users\\HP\\Documents\\datasets\\vehicle.csv")

selected_features = ['COMPACTNESS', 'MAX.LENGTH_ASPECT_RATIO', 'SCALED_VARIANCE_MINOR', 'MAX.LENGTH_RECTANGULARITY']

label_encoder = LabelEncoder()
data['Class'] = label_encoder.fit_transform(data['Class'])

Data_subset = data[selected_features]
target_subset = data['Class']

X_subset = Data_subset
y_subset = target_subset

scaler_subset = StandardScaler()
X_scaled_subset = scaler_subset.fit_transform(X_subset)

X_train_scaled_subset, X_test_scaled_subset, y_train_subset, y_test_subset = train_test_split(X_scaled_subset, y_subset, test_size=0.33)

svm_clf = SVC(kernel='rbf')

svm_clf.fit(X_train_scaled_subset, y_train_subset)

predictions_svm = svm_clf.predict(X_test_scaled_subset)

accuracy_svm = accuracy_score(y_test_subset, predictions_svm)
print("Accuracy with SVM (RBF Kernel):", accuracy_svm)

dump(svm_clf, 'best_svm_rbf_model.joblib')

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import joblib

# read the training data
data_import = pd.read_csv("data_oe_training.csv")

# remove the result from the data
data_labels = data_import.pop("result")

# split the data into train and test
x_train, x_test, y_train, y_test = train_test_split(
    data_import, data_labels, 
    test_size = 0.2, 
    random_state = 42
)

# scale the data
scaler = StandardScaler()
x_train_scale = scaler.fit_transform(x_train)
x_test_scale = scaler.transform(x_test)
joblib.dump(scaler, 'std_scaler_rf.bin', compress = True)

# set up the random forest model
model = RandomForestClassifier(n_estimators = 500, random_state = 42)

# train the model
model.fit(x_train_scale, y_train)

""" 500 trees is a plateau for accuracy
# set up the params for trees
param_grid = {"n_estimators": [10, 50, 100, 200, 500]}

# create the grid search model
grid_search = GridSearchCV(model, param_grid, cv=5)

# train the model
grid_search.fit(x_train_scale, y_train)

print(grid_search.best_params_)
print(grid_search.best_score_)
"""

# predict using the model
prediction = model.predict(x_test_scale)
predict_prob = model.predict_proba(x_test_scale)
score = model.score(x_test_scale, y_test)
print(f"\nNOTE: Score of the overall model: {int(score * 100)}%")

# create a testing mechanism
result_by_class = {}
classes_win = [.8, .7, .5]
classes_loss = [.2, .3, .49]

for pred, prob, result in zip(prediction, predict_prob, y_test):

    for bin in classes_win:
        try:
            result_by_class[bin]
        except:
            result_by_class[bin] = []
        if round(prob[0], 2) >= bin:
            if pred == result:
                result_by_class[bin].append(1)
            else:
                result_by_class[bin].append(0)

    for bin in classes_loss:
        try:
            result_by_class[bin]
        except:
            result_by_class[bin] = []
        if round(prob[0], 2) <= bin:
            if pred == result:
                result_by_class[bin].append(1)
            else:
                result_by_class[bin].append(0)

print("\nNOTE: Predicted Wins")
for bin in classes_win:
    bin_mean = np.average(result_by_class[bin])
    bin_count = len(result_by_class[bin])
    print(f"{bin_count} games at or above {int(100 * bin)}% likelihood (Accuracy: {int(bin_mean * 100)}%)")

print("\nNOTE: Predicted Losses")
for bin in classes_loss:
    bin_mean = np.average(result_by_class[bin])
    bin_count = len(result_by_class[bin])
    if bin == .49:
        print(f"{bin_count} games at or above 50% likelihood (Accuracy: {int(bin_mean * 100)}%)")
    else:
        print(f"{bin_count} games at or above {int(100 * (1 - bin))}% likelihood (Accuracy: {int(bin_mean * 100)}%)")

# save model
joblib.dump(model, "model_random_forest.bin")
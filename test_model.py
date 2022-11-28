from tensorflow import keras
import pandas as pd
import joblib

# bring in the model
model = keras.models.load_model("saved_model/league_oe_data")
normalizer = joblib.load("std_scaler.bin")

# bring in the data to predict
data_predict = pd.read_csv("data_oe_training.csv")
data_predict_result = data_predict.pop("result")

# predict one observation
prediction = model.predict(normalizer.transform(data_predict))

# test to see how good the prediction is for each tranche
def test_predict(input_result, input_prediction, size):
    count_90_total = 0
    count_90_correct = 0
    count_80_total = 0
    count_80_correct = 0
    count_70_total = 0
    count_70_correct = 0
    count_60_total = 0
    count_60_correct = 0
    count_50_total = 0
    count_50_correct = 0
    count_49_total = 0
    count_49_correct = 0
    count_40_total = 0
    count_40_correct = 0
    count_30_total = 0
    count_30_correct = 0
    count_20_total = 0
    count_20_correct = 0
    count_10_total = 0
    count_10_correct = 0

    for i in range(size):
        if input_prediction[i][0] >= .9:
            count_90_total += 1
            if input_result[i] == 1:
                count_90_correct += 1
        if input_prediction[i][0] >= .8:
            count_80_total += 1
            if input_result[i] == 1:
                count_80_correct += 1
        if input_prediction[i][0] >= .7:
            count_70_total += 1
            if input_result[i] == 1:
                count_70_correct += 1
        if input_prediction[i][0] >= .6:
            count_60_total += 1
            if input_result[i] == 1:
                count_60_correct += 1
        if input_prediction[i][0] >= .5:
            count_50_total += 1
            if input_result[i] == 1:
                count_50_correct += 1
        if input_prediction[i][0] < .5:
            count_49_total += 1
            if input_result[i] == 0:
                count_49_correct += 1
        if input_prediction[i][0] <= .4:
            count_40_total += 1
            if input_result[i] == 0:
                count_40_correct += 1
        if input_prediction[i][0] <= .3:
            count_30_total += 1
            if input_result[i] == 0:
                count_30_correct += 1
        if input_prediction[i][0] <= .2:
            count_20_total += 1
            if input_result[i] == 0:
                count_20_correct += 1
        if input_prediction[i][0] <= .1:
            count_10_total += 1
            if input_result[i] == 0:
                count_10_correct += 1
    
    return [
        count_90_total, (count_90_correct / count_90_total), 
        count_80_total, (count_80_correct / count_80_total), 
        count_70_total, (count_70_correct / count_70_total), 
        count_60_total, (count_60_correct / count_60_total), 
        count_50_total, (count_50_correct / count_50_total),
        count_49_total, (count_49_correct / count_49_total), 
        count_40_total, (count_40_correct / count_40_total), 
        count_30_total, (count_30_correct / count_30_total), 
        count_20_total, (count_20_correct / count_20_total), 
        count_10_total, (count_10_correct / count_10_total)
    ]

result_test_prediction = test_predict(data_predict_result, prediction, data_predict_result.shape[0])
print("\nWin predictions from model:")
print(f"  # games >= 90% confidence: {result_test_prediction[0]}")
print(f"    % games predicted correctly: {int(result_test_prediction[1] * 100)}%")
print(f"  # games >= 80% confidence: {result_test_prediction[2]}")
print(f"    % games predicted correctly: {int(result_test_prediction[3] * 100)}%")
print(f"  # games >= 70% confidence: {result_test_prediction[4]}")
print(f"    % games predicted correctly: {int(result_test_prediction[5]* 100)}%")
print(f"  # games >= 60% confidence: {result_test_prediction[6]}")
print(f"    % games predicted correctly: {int(result_test_prediction[7] * 100)}%")
print(f"  # games >= 50% confidence: {result_test_prediction[8]}")
print(f"    % games predicted correctly: {int(result_test_prediction[9] * 100)}%")

print("\nLoss predictions from model:")
print(f"  # games >= 90% confidence: {result_test_prediction[18]}")
print(f"    % games predicted correctly: {int(result_test_prediction[19] * 100)}%")
print(f"  # games >= 80% confidence: {result_test_prediction[16]}")
print(f"    % games predicted correctly: {int(result_test_prediction[17] * 100)}%")
print(f"  # games >= 70% confidence: {result_test_prediction[14]}")
print(f"    % games predicted correctly: {int(result_test_prediction[15]* 100)}%")
print(f"  # games >= 60% confidence: {result_test_prediction[12]}")
print(f"    % games predicted correctly: {int(result_test_prediction[13] * 100)}%")
print(f"  # games > 50% confidence: {result_test_prediction[10]}")
print(f"    % games predicted correctly: {int(result_test_prediction[11] * 100)}%")
import pandas as pd
from tensorflow import keras
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# set a seed for the model
os.environ['PYTHONHASHSEED'] = str(0)
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# take in the data
data_oe_intake = pd.read_csv("data_oe_training.csv")

# Split the data
oe_training_features = data_oe_intake.sample(frac = 0.8)
oe_training_label = oe_training_features.pop("result")
oe_testing_features = data_oe_intake.drop(oe_training_features.index)
oe_testing_label = oe_testing_features.pop("result")

# Normalize the data
normalizer = StandardScaler()
oe_training_features_norm = normalizer.fit_transform(oe_training_features)
oe_testing_features_norm = normalizer.transform(oe_testing_features)
joblib.dump(normalizer, 'std_scaler.bin', compress = True)

# Build the model
def get_model():
    model = keras.models.Sequential([
        keras.layers.Dense(2, input_shape = (48,), kernel_regularizer = keras.regularizers.l2()),
        keras.layers.Dense(1, activation = "sigmoid")
    ])
    model.compile(
        loss = "binary_crossentropy",
        optimizer = "adam",
        metrics = ["accuracy"]
    )
    return model

print("\nNOTE: Model summary:")
get_model().summary()

# Train the model
es_cb = keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 5)

model = get_model()

preds_on_untrained = model.predict(oe_testing_features_norm)

print("\nNOTE: Training begins here:")
history = model.fit(
    oe_training_features_norm, oe_training_label,
    validation_data = (oe_testing_features_norm, oe_testing_label),
    epochs = 100,
    callbacks = [es_cb]
)

# Save model
model.save('saved_model/league_oe_data')

# save down the training and validation loss to check overfitting
train_loss = history.history["loss"]
val_loss = history.history["val_loss"]

# plot the training and validation losses
plt.plot(train_loss, label = "Training loss")
plt.plot(val_loss, label = "Validation loss")
plt.legend()
plt.show()
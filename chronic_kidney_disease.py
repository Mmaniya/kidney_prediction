import glob
from keras.models import Sequential, load_model
import numpy as np
import pandas as pd
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import keras as k 

#uploaded = files.upload()      
df = pd.read_csv("kidney_disease.csv")
    


columns_to_retain = ["sg", "al", "sc", "hemo", "pcv", "wbcc", "rbcc", "htn", "classification"]

df = df.drop([col for col in df.columns if not col in columns_to_retain], axis=1)
    
df = df.dropna(axis=0)

for column in df.columns:
        if df[column].dtype == np.number:
            continue
        df[column] = LabelEncoder().fit_transform(df[column])

df.head()

X = df.drop(["classification"], axis=1)
y = df["classification"]

x_scaler = MinMaxScaler()
x_scaler.fit(X)
column_names = X.columns
X[column_names] = x_scaler.transform(X)

X_train,  X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, shuffle=True)

model = Sequential()
model.add(Dense(256, input_dim=len(X.columns),kernel_initializer=k.initializers.random_normal(seed=13), activation="relu"))
model.add(Dense(1, activation="hard_sigmoid"))

model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=2000, batch_size=X_train.shape[0])

model.save("ckd.model")

plt.plot(history.history["accuracy"])
plt.plot(history.history["loss"])
plt.title("model accuracy & loss")
plt.ylabel("accuracy and loss")
plt.xlabel("epoch")
plt.legend(['acc', 'loss'], loc='lower right')
plt.show()

for model_file in glob.glob("*.model"):
  print("Model file: ", model_file)
  model = load_model(model_file)
  print(X_test)
  pred = model.predict(X_test)
  pred = [1 if y>=0.5 else 0 for y in pred] #Threshold, transforming probabilities to either 0 or 1 depending if the probability is below or above 0.5
  scores = model.evaluate(X_test, y_test)
  print("Original  : {0}".format(", ".join([str(x) for x in y_test])))
  print("Predicted : {0}".format(", ".join([str(x) for x in pred])))
  print("Scores    : loss = ", scores[0], " acc = ", scores[1])

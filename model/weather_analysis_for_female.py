import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.optimizers import SGD

# load dataset
data_frame = pandas.read_csv('weather_data_female_test.csv', header=0)
data_set = data_frame.values
# print(data_set.shape)
X = data_set[:, 0: 5].astype(float)
Y = data_set[:, 5]
# encode label values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# print(encoded_Y)
# convert integers to dummy variables
dummy_Y = np_utils.to_categorical(encoded_Y)
print(dummy_Y)

# def baseline_model():
#     # create model
#     model = Sequential()
#     model.add(Dense(8, input_dim=5, activation='relu'))
#     model.add(Dense(8, activation='sigmoid'))
#     # compile model
#     model.compile(loss='binary_crossentropy',
#                   optimizer='adam', metrics=['accuracy'])
#     model.summary()
#     return model


# estimator = KerasClassifier(build_fn=baseline_model,
#                             epochs=200, batch_size=5, verbose=0)
# kfold = KFold(n_splits=10, shuffle=True)
# results = cross_val_score(estimator, X, dummy_Y, cv=kfold)
# print('baseline: %.2f%% (%.2f%%)' % (results.mean()*100, results.std()*100))

model = Sequential()
model.add(Dense(8, input_dim=5, activation='relu'))
model.add(Dense(8, activation='sigmoid'))
# compile model
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(X, dummy_Y, epochs=200, batch_size=5, verbose=0)

print(model.predict(X[0: 1], batch_size=None, verbose=0, steps=None))
print(dummy_Y[0: 1])
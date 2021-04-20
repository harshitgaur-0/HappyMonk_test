import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras import backend as K
import numpy as np


df = pd.read_csv("BankNote_Authentication.csv")
features = df.loc[:, df.columns != "class"]
label = df["class"]
train_x, x_test, train_y, y_test = train_test_split(features,label,test_size=0.2,random_state=7)
scaler = StandardScaler()
scaler.fit(train_x)
train_x = scaler.transform(train_x)
scaler.fit(x_test)
x_test = scaler.transform(x_test)
model = Sequential()
from keras.layers import LeakyReLU
model.add(Dense(512,activation="relu", input_shape=(4,)))
model.add(Dense(512,activation="relu"))
model.add(Dense(2, activation="softmax"))
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
history=model.fit(train_x,train_y,epochs=2,validation_data=(x_test,y_test))
#evaluating thw model
_, train_acc = model.evaluate(train_x,train_y,verbose=0)
_, test_acc = model.evaluate(x_test,y_test,verbose=0)
# f1-Score
yhat_probs = model.predict(x_test,verbose=0)
yhat_class = model.predict_classes(x_test,verbose=0)
from sklearn.metrics import f1_score
f1 = f1_score(y_test,yhat_class)
print("f1 Score = {}".format(f1))

#plot loss during training
plt.title('Loss')
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='test loss')
plt.legend()
plt.show()
# plot loss function vs epochs
plt.subplot(212)
plt.title('Accuracy')
epochs=[1,2]
plt.plot(history.history['loss'], label='loss')
plt.plot(epochs, label='epochs')
plt.legend()
plt.show()
def get_gradient_func(model):
    grads = K.gradients(model.total_loss, model.trainable_weights)
    inputs = model.model._feed_inputs + model.model._feed_targets + model.model._feed_sample_weights
    func = K.function(inputs, grads)
    return func
for epoch in range(1,3):
    model.fit(train_x,train_y,epochs=epoch,validation_data=(x_test,y_test))
    get_gradient = get_gradient_func(model)
    grads = get_gradient([train_x, train_y, np.ones(len(train_y))])
    print("epochs {} : parameter update on epochs {} ".format(epoch,grads))
    if epoch==2:
        print("Final updated Parameter {}".format(grads))

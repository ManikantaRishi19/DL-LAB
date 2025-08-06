#To implement MLP on MNIST dataset using keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt

#load data
(x_train,y_train),(x_test,y_test)=mnist.load_data()

#preprocesing
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)


#build the architecture
model=Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(units=10,activation='softmax'))

#compile
model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])

#training
res=model.fit(x_train,y_train,epochs=30,batch_size=32,validation_data=(x_test,y_test))
print(res.history.keys())
print(res.history.items())

#evaluate
loss,accuracy=model.evaluate(x_test,y_test)
print(f"test loss:{loss},\n test_accuracy:{accuracy}")

#visualization
plt.plot(res.history['loss'],label='train_loss',color='blue')
plt.plot(res.history['val_loss'],label='validation_loss',color='red')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Epochs vs Loss")
plt.legend()
plt.show()

#for validation accuracy
plt.plot(res.history['accuracy'],label='train_accuracy',color='blue')
plt.plot(res.history['val_accuracy'],label='validation_accuracy',color='red')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Epochs vs Accuracy")
plt.legend()
plt.show()

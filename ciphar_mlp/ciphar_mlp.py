import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.optimizers import Adam
from matplotlib import pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train_cat = to_categorical(y_train)
y_actuals = y_test
y_test_cat = to_categorical(y_test)

model = Sequential()
model.add(Flatten(input_shape=(32, 32, 3)))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

result = model.fit(x_train, y_train_cat, epochs=15, batch_size=64, validation_split=0.2)

test_loss, test_accuracy = model.evaluate(x_test, y_test_cat)

predictions = model.predict(x_test)
predictedLabel = np.argmax(predictions, axis=1)
print(f"Predicted label: {(predictedLabel[10])}")

for image in range(5):
    print(f"Act: {y_test[image]} \n Pred: {predictedLabel[image]}")
    plt.subplot(1,5,image+1, title=f"Actual: {y_actuals[image]} \n Predicted: {predictedLabel[image]}")
    plt.imshow(x_test[image])
    plt.xticks([])
    plt.yticks([])
plt.show()

plt.plot(result.history['loss'],label='train loss',color='blue')
plt.plot(result.history['val_loss'],label='valdation loss',color='red')
# plt.xticks(np.arange(1, 50, 2))
plt.xlabel("Epo"
           "chs")
# plt.yticks(np.arange(1, 50, 2))
plt.ylabel("Loss")
plt.title("Epochs vs Loss")
plt.legend()
plt.show()

plt.plot(result.history['accuracy'],label='train accuracy',color='blue')
plt.plot(result.history['val_accuracy'],label='valdation accuracy',color='red')
# plt.xticks(np.arange(1, 50, 2))
# plt.yticks(np.arange(1, 50, 2))
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Epochs vs Accuracy")
plt.legend()
plt.show()
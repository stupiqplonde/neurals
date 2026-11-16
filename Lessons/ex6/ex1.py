import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.datasets import mnist # библтоьека выборок мнист

# загрузка обучающей и тестовой выборок
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# нормализация входных данных
x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

size_val = 10000 # размер выборки валидации
x_val_split = x_train[:size_val] # выделение первых наблюдений из обучающей выборки
y_val_split = y_test_cat[:size_val] # в выборку валидации

x_train_split = x_train[size_val:] # выделение последующих наблюдений для обучающей выборки 
y_train_split = y_train_cat[size_val:]

# формирование модель НС и вывод ее структуры в консоль
model = keras.Sequential([
    Input(shape=(28, 28, 1)),   # явный вход
    Flatten(),                  # (28,28,1) -> (784,)
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

myAdam = keras.optimizers.Adam(learning_rate=0.1)
myOpt = keras.optimizers.SGD(learning_rate=0.1, momentum=0.0, nesterov=True)

model.compile(
    optimizer=myOpt,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# запуск процесса обучения (80% - обучающая выборка, 20% - выборка валидации)
model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)

model.evaluate(x_test, y_test_cat)


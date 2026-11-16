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

# отображение первых 25 изображений из обучающей выборки
plt.figure(figsize=(10, 5))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap=plt.cm.binary)

plt.show()

# формирование модель НС и вывод ее структуры в консоль
model = keras.Sequential([
    Input(shape=(28, 28, 1)),   # явный вход
    Flatten(),                  # (28,28,1) -> (784,)
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

print(model.summary()) # вывод структуры в консоль

# компиляция НС с оптимизацией по Adam и критерием - категориальная кросс-энтропия
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# запуск процесса обучения (80% - обучающая выборка, 20% - выборка валидации)
model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)

model.evaluate(x_test, y_test_cat)

# проверка распознавания цифр
# n = 2
# x = np.expand_dims(x_test[n], axis=0)
# res = model.predict(x)
# print(res)
# print(f"Распознаная цифра - {np.argmax(res)}")

# plt.imshow(x_test[n], cmap=plt.cm.binary)
# plt.show()

# распознавание всей тестовой выборки
pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)

print(pred.shape)

print(pred[:20])
print(y_test[:20])

# выделение неверных резов
mask = pred == y_test
print(mask[:10])

x_false = x_test[~mask]
p_false = pred[~mask]

print(x_false.shape)

for i in range(5):
    print("Значение сети: "+str(p_false[i]))
    plt.imshow(x_false[i], cmap=plt.cm.binary)
    plt.show()
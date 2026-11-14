import numpy as np

# Функция активации
def act(x):
    return 0 if x <= 0 else 1


def go(C):
    # Входной вектор с добавленным смещением (bias = 1)
    x = np.array([C[0], C[1], 1])

    # Веса скрытого слоя (2 нейрона, 3 входа)
    w1 = [1, 1, -1.5]  # Нейрон 1
    w2 = [1, 1, -0.5]  # Нейрон 2
    w_hidden = np.array([w1, w2])

    # Веса выходного слоя (1 нейрон, 3 входа: 2 от скрытого + bias)
    w_out = np.array([-1, 1, -0.5])

    # Прямой проход: скрытый слой
    sum = np.dot(w_hidden, x)
    out = [act(x) for x in sum]
    out.append(1)
    out = np.array(out)

    sum = np.dot(w_out, out)
    y = act(sum)
    return y


C1 = [(1, 0), (0, 1)]
C2 = [(0, 0), (1, 1)]

print(go(C1[0]), go(C1[1]))
print(go(C2[0]), go(C2[1]))
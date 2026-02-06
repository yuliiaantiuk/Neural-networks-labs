import tensorflow as tf
import numpy as np
import random

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class EpochLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 100 == 0:
            print(f"Епоха {epoch+1}: loss={logs['loss']:.4f}, accuracy={logs['accuracy']:.4f}")

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

X = np.array([
    [0,0,0,0],
    [0,0,0,1],
    [0,0,1,0],
    [0,0,1,1],
    [0,1,0,0],
    [0,1,0,1],
    [0,1,1,0],
    [0,1,1,1],
    [1,0,0,0],
    [1,0,0,1],
    [1,0,1,0],
    [1,0,1,1],
    [1,1,0,0],
    [1,1,0,1],
    [1,1,1,0],
    [1,1,1,1]
])

y = np.array([
    0,
    1,
    1,
    0,
    1,
    0,
    0,
    1,
    1,
    0,
    0,
    1,
    0,
    1, 
    1, 
    0   
])


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(8, activation=tf.nn.leaky_relu),
    tf.keras.layers.Dense(4, activation=tf.nn.leaky_relu),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(X, y, epochs=1500, verbose=0, callbacks=[EpochLogger()])

predictions = model.predict(X)
predictions_rounded = np.round(predictions).astype(int)

print(" x1 x2 x3 x4 | y_true | y_pred | is correct?")
print("--------------------------------------")

for i in range(len(X)):
    correct = "+" if predictions_rounded[i][0] == y[i] else "-"
    print(
        f" {X[i][0]}  {X[i][1]}  {X[i][2]}  {X[i][3]}  |"
        f"   {y[i]}    |   {predictions_rounded[i][0]}    | {correct}"
    )

print("\nWeights and biases")

for i, layer in enumerate(model.layers):
    if isinstance(layer, tf.keras.layers.Dense):
        w, b = layer.get_weights()
        print(f"\nLayer {i}: {layer.name}")
        print("Weights:")
        print(w)
        print("Biases:")
        print(b)


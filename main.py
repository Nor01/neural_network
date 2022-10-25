+import tensorflow
import numpy

celsius = numpy.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = numpy.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

# layer = tensorflow.keras.layers.Dense(units=1, input_shape=[1])
# model = tensorflow.keras.Sequential([layer])

# model.compile(
#     optimizer=tensorflow.keras.optimizers.Adam(0.1),
#     loss='mean_squared_error'
# )

# print("Training starts")

# all_history = model.fit(celsius, fahrenheit, epochs=10000, verbose=False)

# print("Model trained")

# print("Prediction Starts!")
# result = model.predict([100.0])
# print(f"The result is: {str(result)} farenheit!")

# # INPUT: Adam 0.1, epochs=1000
# # #1 The result is: [[211.74399]] farenheit!
# # Really close, because 100 C = 212 F

# # #2 The result is: [[211.7414]] farenheit!

# # INPUT: Adam 0.1, epochs=5000
# # #3 The result is: [[211.74744]] farenheit!

# # INPUT: Adam 0.1, epochs=10000
# # #4 The result is: [[211.75784]] farenheit!

# # INPUT: Adam 0.1, epochs=100000
# # #5 The result is: [[211.74754]] farenheit!
# # No much change comparing to epochs=10000 however the internal values

# print("Internal model's values")
# print(layer.get_weights())

# # The result is:
# # [array([[1.7980396]], dtype=float32), array([31.952538], dtype=float32)]

# # Something amazing because our formula to calculate C to F is:
# # F = C * 1.8 + 32
# # F = 100.0 * 1.7980396 + 31.952538

# # Fantastic! For 1 Layer and 1 Neural

# What happen if I add more Layers and Neurals?

hide_layer1 = tensorflow.keras.layers.Dense(units=3, input_shape=[1])
hide_layer2 = tensorflow.keras.layers.Dense(units=3)
output_layer = tensorflow.keras.layers.Dense(units=1)
model = tensorflow.keras.Sequential([hide_layer1,hide_layer2,output_layer])

model.compile(
    optimizer=tensorflow.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print("Training starts")
all_history = model.fit(celsius, fahrenheit, epochs=10000, verbose=False)
print("Model trained")

print("Prediction Starts!")
result = model.predict([100.0])
print(f"The result is: {str(result)} farenheit!")

# #6 The result is: [[211.74776]] farenheit!

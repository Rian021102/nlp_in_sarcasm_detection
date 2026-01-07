import tensorflow as tf

print("Available devices:")
print(tf.config.list_physical_devices())

print("GPU devices:")
print(tf.config.list_physical_devices('GPU'))

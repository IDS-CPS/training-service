import tensorflow as tf

class Encoder(tf.keras.layers.Layer):
  def __init__(self, intermediate_dim=32):
    super(Encoder, self).__init__()
    self.flatten = tf.keras.layers.Flatten()
    self.hidden_layer = tf.keras.layers.Dense(
      units=intermediate_dim,
      activation=tf.nn.tanh
    )
    self.output_layer = tf.keras.layers.Dense(
      units=intermediate_dim,
      activation=tf.nn.tanh
    )
    
  def call(self, input_features):
    flatten = self.flatten(input_features)
    activation = self.hidden_layer(flatten)
    return self.output_layer(activation)

class Decoder(tf.keras.layers.Layer):
  def __init__(self, original_dim, intermediate_dim=32):
    super(Decoder, self).__init__()
    self.hidden_layer = tf.keras.layers.Dense(
      units=intermediate_dim,
      activation=tf.nn.tanh
    )
    self.output_layer = tf.keras.layers.Dense(
      units=original_dim,
      activation=tf.nn.tanh
    )
  
  def call(self, code):
    activation = self.hidden_layer(code)
    return self.output_layer(activation)

class Autoencoder(tf.keras.Model):
  def __init__(self, train_shape):
    super(Autoencoder, self).__init__()
    self.train_shape = train_shape
    self.encoder = Encoder(intermediate_dim=0.5*train_shape[1]*train_shape[2])
    self.decoder = Decoder(original_dim=train_shape[1]*train_shape[2], intermediate_dim=0.5*train_shape[1]*train_shape[2])
  
  def call(self, input_features):
    code = self.encoder(input_features)
    decoded = self.decoder(code)
    output = tf.reshape(decoded, [-1, self.train_shape[1], self.train_shape[2]])
    return output
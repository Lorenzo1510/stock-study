from typing import Tuple
import tensorflow as tf


class FeedBack(tf.keras.Model):
    def __init__(self, units: int, out_steps: int, num_features: int):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(num_features)

    def warmup(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, list]:
        x, *state = self.lstm_rnn(inputs)
        prediction = self.dense(x)
        return prediction, state

    def call(self, inputs: tf.Tensor, training=None) -> tf.Tensor:
        predictions = []
        prediction, state = self.warmup(inputs)
        predictions.append(prediction)
        
        for _ in range(1, self.out_steps):
            x = prediction
            x, state = self.lstm_cell(x, states=state, training=training)
            prediction = self.dense(x)
            predictions.append(prediction)
        
        predictions = tf.stack(predictions)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions

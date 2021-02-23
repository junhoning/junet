import tensorflow as tf


# Dice Coefficient
def precision(y_true, y_pred):
    axes = tuple(range(1, len(y_pred.shape)-1))
    
    true_positives = tf.reduce_sum(tf.math.round(tf.clip_by_value(y_true * y_pred, 0, 1)), axes)
    predicted_positives = tf.reduce_sum(tf.math.round(tf.clip_by_value(y_pred, 0, 1)), axes)
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    
    return tf.reduce_mean(precision)


def recall(y_true, y_pred):
    axes = tuple(range(1, len(y_pred.shape)-1))
    
    true_positives = tf.reduce_sum(tf.math.round(tf.clip_by_value(y_true * y_pred, 0, 1)), axes)
    possible_positives = tf.reduce_sum(tf.math.round(tf.clip_by_value(y_true, 0, 1)), axes)
    recall = true_positives / (possible_positives +  tf.keras.backend.epsilon())
    
    return tf.reduce_mean(recall)

def dice(y_true, y_pred):
    epsilon=tf.keras.backend.epsilon()
    axes = tuple(range(1, len(y_pred.shape)-1)) 
    numerator = 2. * tf.reduce_sum(y_pred * y_true, axes)
    denominator = tf.reduce_sum(tf.math.square(y_pred) + tf.math.square(y_true), axes)
    
    return tf.reduce_mean(numerator / (denominator + epsilon))

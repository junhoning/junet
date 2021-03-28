import tensorflow as tf
import tensorflow_addons as tfa


class Losses:
    def __init__(self, loss_name):
        self.loss_name = loss_name
    
    def loss_function(self, y_true, y_pred):
        if self.loss_name == 'soft_dice_loss':
            return self.soft_dice_loss(y_true, y_pred)
        elif self.loss_name == 'bce_dice_loss':
            return self.bce_dice_loss(y_true, y_pred)
        elif self.loss_name == 'bce_logdice_loss':
            return self.bce_logdice_loss(y_true, y_pred)

    def soft_dice_loss(self, y_true, y_pred):
        epsilon=tf.keras.backend.epsilon()
        axes = tuple(range(1, len(y_pred.shape)-1)) 
        numerator = 2. * tf.reduce_sum(y_pred * y_true, axes)
        denominator = tf.reduce_sum(tf.math.square(y_pred) + tf.math.square(y_true), axes)
        
        return 1-tf.reduce_mean(numerator / (denominator + epsilon))

    def bce_dice_loss(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        return tf.keras.losses.CategoricalCrossentropy(y_true, y_pred) + self.soft_dice_loss(y_true, y_pred)

    def bce_logdice_loss(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        return tf.keras.losses.CategoricalCrossentropy(y_true, y_pred) - tf.math.log(1. - self.soft_dice_loss(y_true, y_pred))


class Optimizers:
    def __init__(self, optm_name='adam', optm_avg_name=None):
        self.optm_name = optm_name
        self.optm_avg_name = optm_avg_name
    
    def optm(self, learning_rate):
        if self.optm_name == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        elif self.optm_name == 'rmsprop':
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate)
        else:
            raise "Choose from 'adam, rmsprop'"

    def optm_avg(self, optm):
        '''
        이동 평균 : 이동 평균화의 장점은 최신 배치에서 급격한 손실 이동이나 불규칙한 데이터 표현에 덜 취약하다는 것입니다. 
                어느 시점까지 모델 훈련에 대한 좀 더 일반적인 아이디어를 제공합니다.
        확률 적 평균 : 확률 적 가중치 평균은 더 넓은 최적 값으로 수렴됩니다. 
                이렇게하면 기하학적 앙상블 링과 비슷합니다. 
                SWA는 다른 옵티 마이저 주위의 래퍼로 사용될 때 모델 성능을 향상시키고 
                내부 옵티마이 저의 다른 궤적 지점에서 결과를 평균화하는 간단한 방법입니다.
        '''
        if self.optm_avg_name == 'moving':
            return tfa.optimizers.MovingAverage(optm)
        elif self.optm_avg_name == 'stochastic':
            return tfa.optimizers.SWA(optm)

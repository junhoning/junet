import tensorflow as tf
import os

def save_ckpt(checkpoint_dir, step, sess, name):
    model_name = "fusionNet.model"
    model_dir = name
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    tf.train.Saver().save(sess, os.path.join(checkpoint_dir, model_name),
                          global_step=step)


def load_ckpt(checkpoint_dir, sess, name):
    print(" [*] Reading checkpoint...")

    model_dir = name
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        tf.train.Saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False
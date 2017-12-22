import argparse
import time
from datetime import datetime

import numpy as np
import tensorflow as tf


class Trainer(object):
    def __init__(self, project_title='temp', work_dir='c://junet_workspace/', data_ext='tfrecords', num_gpus=None,
                 tboard_record=True, **train_data):
        self.project_title = project_title
        self.work_dir = work_dir
        self.num_gpus = num_gpus
        self.data_ext = data_ext
        self.tboard_record = tboard_record
        print('Segmentation initial loaded')

    def train(self, subject_title='temp', num_epochs=50):
        with tf.Session(config=self.sess_config) as sess:
            sess.run(self.init_op)

            summary_writer = tf.summary.FileWriter(self.work_dir, sess.graph)

            for epoch in range(num_epochs):
                for step in range(self.batches_per_epoch):
                    start_time = time.time()
                    _, loss_value = sess.run([train_op, self.loss])
                    duration = time.time() - start_time

                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                    if step % 10 == 0:
                        num_examples_per_step = self.batch_size * self.num_gpus
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = duration / self.num_gpus

                        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                      'sec/batch)')
                        print(format_str % (datetime.now(), step, loss_value,
                                            examples_per_sec, sec_per_batch))

                    if step % 100 == 0:
                        summary_str = sess.run(self.summary_op)
                        summary_writer.add_summary(summary_str, step)

                    # Save the model checkpoint periodically.
                    if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                        checkpoint_path = os.path.join(self.work_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_gpus', type=str, default=None,
                        help='Number of GPUs')

    parser.add_argument('--project_title', type=str, default='SegEngine',
                        help='Title of Project')

    parser.add_argument('--batch_size', type=int, default=4,
                        help='Number of Images to process in Batch')

    parser.add_argument('--grid_n', type=int, default=1,
                        help='Number of Grid for cropping images')

    parser.add_argument('--input_shape', nargs='+', type=int, default=(64, 64, 64),
                        help='Input Size You want to crop from the Original Size')

    parser.add_argument('--data_ext', type=str, default='tfrecords',
                        help='File Ext of the target data file')

    parser.add_argument('--aug_list', nargs='+', type=str, default='',
                        help='Augmentation List')

    parser.add_argument('--inbound_shape', nargs='+', type=int, default=parser.parse_args().input_shape,
                        help='Inbound Range to Limit the Grid from Original Data')

    parser.add_argument('--log_device_placement', type=bool, default=False,
                        help='Whether to log device placement')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Initial Learning Rate')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of Epochs')
    parser.add_argument('--model_name', type=str, default='segmentation',
                        help='Name of Model')
    parser.add_argument('--loss_name', type=str, default='dice',
                        help='Name of Loss Function')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Name of Optimization')
    parser.add_argument('--num_epochs_per_decay', type=int, default=3,
                        help='Number of Epochs per Decay')
    parser.add_argument('--lr_decay_factor', type=float, default=1,
                        help='Learning Rate decay factor')

    parser.add_argument('--loss-name', type=str, default='ce',
                        help='Name of Loss Name')

    FLAGS = parser.parse_args()

    params = {'grid_n': FLAGS.grid_n,
              'input_shape': FLAGS.input_shape,
              'inbound_shape': FLAGS.inbound_shape,
              'lr_decay_factor': FLAGS.lr_decay_factor,
              'num_epochs_per_decay': FLAGS.num_epochs_per_decay,
              'weight_decay': FLAGS.weight_decay,
              'data_ext': FLAGS.data_ext,
              'work_dir': FLAGS.work_dir,
              'data_dir': FLAGS.data_dir,
              'num_gpus': FLAGS.num_gpus,
              'log_device_placement': FLAGS.log_device_placement}

    train(learning_rate=FLAGS.learning_rate, num_epochs=FLAGS.num_gpus, batch_size=FLAGS.batch_size,
          project_title=FLAGS.project_title, optimizer=FLAGS.optimizer, loss_name=FLAGS.loss_name,
          aug_list=FLAGS.aug_list, model_name=FLAGS.model_name, **params)
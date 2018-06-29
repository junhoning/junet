from datetime import datetime
import tensorflow as tf

from ..utils.generic_utils import progress_bar
from .reporter import save_ckpt


def training_loops(scope, dataset, image, label, save_dir, epochs, train_op):
    init_group = tf.group([tf.global_variables_initializer(), tf.local_variables_initializer()])

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver(max_to_keep=3)

    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
    summary_op = tf.summary.merge(summaries)

    with tf.Session(config=config) as sess:
        # 학습을 시작하기 전에 Variable 들을 초기화 한다.
        sess.run(init_group)
        summary_writer = tf.summary.FileWriter(save_dir, sess.graph)
        print("Start Training!", datetime.now())
        glob_step = 0
        eval_step = 0
        epoch = 0
        if epochs == None:
            epochs = float("inf")

        while epochs >= epoch:
            avg_cost = 0.
            avg_accr = 0.

            for batch_n in range(dataset.batches_per_epoch):
                dataset.generator(batch_n)
                train_feed = {image: dataset.batch_image, label: dataset.batch_label}
                sess.run(train_op, train_feed)

                # accr_value, _ = sess.run(update_op + [accr], train_feed)

                metric_collect = tf.get_collection('metric_ops', scope)
                accr_value, cost_value = sess.run(metric_collect, train_feed)
                avg_cost += cost_value
                avg_accr += accr_value

                progress_bar(dataset.batches_per_epoch, batch_n + 1,
                             state_msg='Step : %d/%d  Average Cost : %.4f, '
                                       'Average Accuracy : %.4f' % (batch_n, epoch,
                                                                    avg_cost / (batch_n + 1),
                                                                    avg_accr / (batch_n + 1),
                                                                    ))

                glob_step += 1

                if batch_n % 5 == 0 and batch_n > 0:
                    # saver.save(sess, checkpoint_dir + str(glob_step))
                    summary_str = sess.run(summary_op, train_feed)
                    summary_writer.add_summary(summary_str, glob_step)

            epoch += 1
            if epoch % 1 == 0:
                save_ckpt(sess, saver, save_dir, glob_step)

        print("Done training at : ", datetime.now(), "\n")

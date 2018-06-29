import os
from datetime import datetime
import tensorflow as tf
import numpy as np


def training_loop(train_params, train_op, init_op, sess_config, checkpoint_dict, handle):
    print('Number of Datas : Train %d / Validation %d' % (len(train_list), len(valid_list)))
    print("Start Training at ", datetime.now())
    with tf.Session(config=sess_config) as sess:
        sess.run(init_op)
        summary_writer = tf.summary.FileWriter(['checkpoint_dir'], sess.graph)
        eval_summary_writer = tf.summary.FileWriter(os.path.join(checkpoint_dict['checkpoint_dir'], 'eval/'),
                                                    sess.graph)

        train_handle = sess.run(train_iterator.string_handle())
        valid_handle = sess.run(valid_iterator.string_handle())
        glob_step = -1
        for epoch in range(num_epochs):
            sess.run(valid_iterator.initializer)
            loss_sum = 0
            avg_step = 0
            accr_sum = 0
            for step in range(batches_per_epoch):
                glob_step += 1
                train_feed = {handle: train_handle}
                sess.run(train_op, feed_dict=train_feed)

                # Write Report State and Print Loss and Accuracy
                if step in np.linspace(1, batches_per_epoch - 1, report_per_epoch).astype(int):
                    state = {}
                    if verbosity > 1:
                        loss_value, accr_value = sess.run([loss, accr], train_feed)
                        loss_sum += loss_value
                        accr_sum += accr_value
                        avg_step += 1
                        # 'dice': dice_sum/avg_step, 'test_dice': test_dice_sum/avg_step,
                        state['cost'] = loss_sum / avg_step
                        state['accr'] = accr_sum / avg_step

                    reports.report_state(state, glob_step, num_epochs, epoch, batches_per_epoch, step)

                    summary_str = sess.run(summary_op, train_feed)
                    summary_writer.add_summary(summary_str, glob_step)

                if step in np.linspace(0, batches_per_epoch - 1, valid_per_epoch + 1)[1:].astype(int):
                    valid_feed = {handle: valid_handle, is_training: False}
                    _, _, eval_summary = sess.run([loss, accr, summary_op], valid_feed)
                    eval_summary_writer.add_summary(eval_summary, glob_step)
                    # loss_value, accr_value = sess.run([loss, accr], valid_feed)

                # if step in np.linspace(0, batches_per_epoch-1, save_per_epoch+1)[1:].astype(int):
                #     saver.save(sess, checkpoint_path, 0)

                if step in np.linspace(0, batches_per_epoch - 1, evals_per_epoch + 1)[1:].astype(int):
                    print("\nStart Evaluating")
                    sess.run(valid_iterator.initializer)

                    import scipy.misc
                    save_path = reports.save_path
                    valid_feed = {handle: valid_handle, is_training: False}
                    target, groundtruth, logit_result = sess.run([images[i], labels[i], logits], valid_feed)
                    result = np.argmax(logit_result, 3)
                    label = np.argmax(groundtruth, 3)
                    for n in range(8):
                        scipy.misc.toimage(target[n, :, :, 0], cmin=0, cmax=255).save(
                            os.path.join(save_path, 'image_%d_%d.jpg' % (epoch, n)))
                        scipy.misc.toimage(result[n], cmin=0, cmax=1).save(
                            os.path.join(save_path, 'result_%d_%d.jpg' % (epoch, n)))
                        scipy.misc.toimage(label[n], cmin=0, cmax=1).save(
                            os.path.join(save_path, 'label_%d_%d.jpg' % (epoch, n)))

        print("Done training at : ", datetime.now(), "\n")


def training_loops2():
    train_batches_per_epoch = batch_size // len(train_list)
    with tf.Session() as sess:
        sess.run(init_group)
        summary_writer = tf.summary.FileWriter(checkpoint_dir, sess.graph)
        print("Start Training!", datetime.now())
        eval_step = 0
        for epoch in range(epochs):
            avg_cost = 0.
            avg_accr = 0.

            np.random.shuffle(train_list)
            for batch_n in range(train_batches_per_epoch):
                image_batch, label_batch = self.input_data(batch_n)

                train_feed_dict = {image: image_batch, label: label_batch, keep_prob: self.keep_prob}
                sess.run(optm, feed_dict=train_feed_dict)

                # Cost 평균값 업데이트
                cost_value, accr_value = sess.run([cost, accr], feed_dict=train_feed_dict)
                avg_cost += cost_value / self.train_batches_per_epoch
                avg_accr += accr_value / self.train_batches_per_epoch

                progress_bar(self.train_batches_per_epoch, batch_n,
                             state_msg='Average Cost : %.4f, Average Accuracy : %.4f' % (avg_cost, avg_accr))

                self.glob_step += 1

                if batch_n % 10 == 0 and batch_n > 0:
                    summary_str = sess.run(summary_op, train_feed_dict)
                    summary_writer.add_summary(summary_str, self.glob_step)

            # Display logs per epoch step
            if epoch % self.display_step == 0:
                eval_batch_n = (len(self.train_list) * eval_step) % len(self.test_list)
                test_image_batch, test_label_batch = self.input_data(eval_batch_n, mode='test')
                test_feed_dict = {image: test_image_batch, label: test_label_batch, keep_prob: self.keep_prob,
                                  is_training: False}
                print("Epoch: %03d/%03d, Cost: %.9f" % (epoch + 1, self.train_epochs, avg_cost))
                train_acc = sess.run(accr, feed_dict=train_feed_dict)
                print(" Training Accuracy: %.3f" % (train_acc))
                test_acc = sess.run(accr, feed_dict=test_feed_dict)
                print(" Test Accuracy: %.3f" % (test_acc))
                eval_step += 1

            save_ckpt(sess, self.glob_step)

        print("Done training at : ", datetime.now(), "\n")
import configparser
from datetime import datetime

import tensorflow as tf
from JuneNet.model.Segmentation import FusionNet_3D, UNet_3D, UNet_2D, FusionNet_2D
from numpy import linspace

from JuneNet.model.builder import evaluate, loss
from JuneNet.process import reporter

config = configparser.ConfigParser()
config.read_file(open("config/configuration.txt"))
config.add_section('data stats')

optimizer_name = tf.train.AdamOptimizer
loss_name = loss.hybrid_cost


def train_data(train_setting):
    '''
    :param train_setting:
    ## Default ##
    data_type = 'tfrecords' / 'hdf5'
    remove_label = []
    grid_n = 1
    inbound_shape = img_shape
    :return:
    '''


    data_dimension = train_setting['train_dimension']
    if data_dimension == 3:
        train_model = FusionNet_3D
        UNet = UNet_3D
        data_manager = data_manager_3d
    else:
        train_model = FusionNet_2D
        UNet = UNet_2D
        data_manager = data_manager_2d

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default() as graph:
        is_training = tf.placeholder_with_default(True, shape=[], name='is_training')
        image = tf.placeholder(tf.float32, [None] + [img_size] * data_dimension + [num_channel], "input")

        model = model_name.build_model(train_setting, is_training)

        with tf.variable_scope("logit"):
            logit = model.inference(image)
            softmax = tf.nn.softmax(logit, dim=-1, name='softmax_output')

        lbl_size = logit.get_shape().as_list()[1]
        num_class = logit.get_shape().as_list()[-1]
        train_setting['lbl_size'] = lbl_size
        label = tf.placeholder(tf.float32, [None] + [lbl_size] * data_dimension + [num_class], "output")
        data_input = data_manager.Manager(config, train_setting)
        print("Output Shape :", label.get_shape())

        with tf.variable_scope("cost"):
            cost = loss.loss_dice_coef(logit, label)
            if train_setting['record_tboard']:
                tf.summary.scalar("cost", cost)

        with tf.variable_scope("corr"):
            predmax = tf.argmax(logit, data_dimension+1)
            ymax = tf.argmax(label, data_dimension+1)
            corr = tf.equal(predmax, ymax)

        with tf.variable_scope("evaluation"):
            accr = tf.reduce_mean(tf.cast(corr, 'float'))
            dice = evaluate.eval_dice_coef(logit, label)

            if train_setting['record_tboard']:
                tf.summary.scalar('accuracy', accr)
                tf.summary.scalar('dice', dice)

        with tf.variable_scope("Optimizer"):
            # Optimizer
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            var_list = tf.trainable_variables()
            lr = tf.train.exponential_decay(train_setting['learning_rate'],
                                            global_step,
                                            train_setting['decay_step'],
                                            train_setting['learning_rate_decay_factor'],
                                            staircase=True)
            # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(lr)
            grads_and_vars = optimizer.compute_gradients(loss=cost, var_list=var_list)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step)
            print("Optimization Ready")

        # The op for initializing the variables.
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Save Report Main to txt File
        reports = reporter.Reporter(train_setting, config, model)

        # Merge all Summaries for TFBoard
        # summary = reports.tfboard_summary_merge_all(record_tboard=train_setting['record_tboard'])

        # Full Usage of GPUs
        # config_gpu = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config_gpu = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config_gpu.gpu_options.allow_growth = True

        # Create Session to start Training
        with tf.Session(config=config_gpu, graph=graph) as sess:
            sess.run(init_op)

            # Load Graph if Graph Exists
            # reporter.load_ckpt(sess, train_setting['project_title'])
            # reporter.save_ckpt(sess, train_setting['subject_title'], 0)
            # reporter.save_pb(sess, train_setting['subject_title'], logit, softmax)

            # Start Training
            print("Training Start at :", datetime.now())
            # Initialize Step and Loss Averages
            glob_step = 0
            for epoch in range(train_setting['num_epochs']):
                loss_sum = 0; dice_sum = 0; test_dice_sum = 0; avg_step = 0; accr_sum =0
                batch_num_per_epoch = data_input.data_num // train_setting['batch_size']

                for batch_n in range(batch_num_per_epoch):
                    batch_image, batch_label, train_img_info = data_input.batch_generate(batch_n, mode='train')
                    train_feed = {image: batch_image, label: batch_label}
                    sess.run(train_op, train_feed)

                    # Write Report State and Print Loss and Accuracy
                    if batch_n in linspace(1, batch_num_per_epoch-1, train_setting['report_per_epoch']).astype(int):
                        loss_value, dice_value, accr_value = sess.run([cost, dice, accr], train_feed)
                        test_image, test_label, test_img_info = data_input.batch_generate(avg_step, mode='test')
                        test_feed = {image: test_image, label: test_label, is_training: False}
                        test_dice_value = sess.run(dice, test_feed)

                        if not loss_value < 10:
                            print('\nNaN accured')
                            break

                        loss_sum += loss_value
                        dice_sum += dice_value
                        test_dice_sum += test_dice_value
                        avg_step += 1
                        accr_sum += accr_value

                        state = {'cost': loss_sum/avg_step, 'dice': dice_sum/avg_step, 'test_dice': test_dice_sum/avg_step, 'accr':accr_sum/avg_step}
                        reports.report_state(state, glob_step, train_setting['num_epochs'], epoch, batch_num_per_epoch, batch_n)

                    glob_step += 1

                    if batch_n in linspace(0, batch_num_per_epoch-1, train_setting['display_per_epoch']+1)[1:].astype(int):
                        # Display Evaluation of Images
                        image_eval, pred_maxout, y_maxout = sess.run([image, predmax, ymax], train_feed)
                        test_img, test_pred_maxout, test_y_maxout = sess.run([image, predmax, ymax], test_feed)
                        for i in range(1):
                            print("\nTrain Image")
                            evaluate.visualize(config, train_img_info, image_eval, pred_maxout, y_maxout,
                                               glob_step, index=i)
                            print("Test Image")
                            evaluate.visualize(config, test_img_info, test_img, test_pred_maxout, test_y_maxout,
                                               glob_step, index=i)

                        reporter.save_ckpt(sess, train_setting['project_title'], 0)
                        # reporter.save_ckpt(sess, train_setting['subject_title'], 0)
                        # reporter.save_pb(sess, train_setting['project_title'], logit, softmax)

                if not loss_value < 10:
                    print('NaN accured')
                    break
                # Save Checkpoint
                # if epoch % saver_step == 0:
                #     reporter.save_ckpt(sess, train_setting['subject_title'], 0)
                #     reporter.save_pb(sess, train_setting['subject_title'], logit)

            # print("Test image")
            for _ in range(2):
                image_eval, pred_maxout, y_maxout = sess.run([image, predmax, ymax], train_feed)
                # for i in range(train_setting['batch_size']):
                #     evaluate.visualize(config, test_img_info, image_eval, pred_maxout, y_maxout, glob_step, index=i)

            print("Done training at : ", datetime.now(), "\n")


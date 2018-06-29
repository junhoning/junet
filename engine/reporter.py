import os
import shutil
from datetime import datetime
import pandas as pd
from numpy import linspace

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.contrib.slim import get_variables_to_restore

from utils.generic_utils import progress_bar

model_name = "SegNet"
log_path = 'log/'
ckpt_dir = 'save_point/'
img_result_dir = 'image/'
result_dir = 'result/'


def save_ckpt(sess, saver, checkpoint_dir, step):
    # checkpoint_dir = os.path.join(os.getcwd(), ckpt_dir)
    # if not os.path.exists(os.path.join(checkpoint_dir, model_dir)):
    #     # Checkpoint save path
    #     os.makedirs(os.path.join(checkpoint_dir, model_dir))

    # save_name = saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)
    save_name = saver.save(sess, os.path.join(checkpoint_dir, 'checkpoint', model_name+'.model'), global_step=step)
    tf.train.write_graph(sess.graph_def, os.path.join(checkpoint_dir, 'checkpoint'), model_name + '.pbtxt', as_text=True)
    print("Check Point Saved : ", save_name)


def load_ckpt(sess, model_dir):
    print(" [*] Reading checkpoint...")
    checkpoint_dir = os.path.join(ckpt_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        variables_to_restore = get_variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        print("Checkpoint is Loaded!!!! : ", os.path.join(checkpoint_dir, ckpt_name))
    else:
        print("No Checkpoint to load...")


# WARNING!! : To save pb file, the Mode has to be in TEST MODE!!
def save_pb(sess, model_dir, *output_nodes):
    # Save '.pbtxt' file
    checkpoint_dir = os.path.join(os.getcwd(), ckpt_dir, model_dir)
    tf.train.write_graph(sess.graph_def, checkpoint_dir, model_dir + '.pbtxt', as_text=True)

    output_list = []
    # it must be the name of 'logit' or 'softmax' that you want to predict
    for output_node in output_nodes:
        # Save '.pb' file
        output_node_names = output_node.name.split(':')[0]
        output_list.append(output_node_names)

    # Rename Input and Output
    # sess.graph.graph_def.node[0].name = "input"

    print("Output Node :", output_list)
    input_graph_def = sess.graph.as_graph_def()
    output_graph_name = ckpt_dir + model_dir + '/' + model_dir + ".pb"

    print("Exporting graph...")
    output_graph_def = graph_util.convert_variables_to_constants(sess, input_graph_def, output_list)

    with tf.gfile.GFile(output_graph_name, "wb") as f:
        f.write(output_graph_def.SerializeToString())


def load_pb(save_name):
    ckpt_path = ckpt_dir + save_name
    pb_file = save_name + '.pb'

    with tf.gfile.GFile(ckpt_path + '/' + pb_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        graph_def.ParseFromString(f.read())

        # fix nodes
        for node in graph_def.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr:
                    del node.attr['use_locking']
            # Warn of this below codes!!!!!!!!!!!!!!!!!!!!!
            elif node.op == 'AssignAdd':
                node.op = 'Sub'
                if 'use_locking' in node.attr:
                    del node.attr['use_locking']

        # for node in graph_def.node[:20]:
        #     print(node.name)
        #
        # for node in graph_def.node[-20:]:
        #     print(node.name)

        tf.import_graph_def(
            graph_def,
            name='prefix',
            input_map=None,
            return_elements=None,
            op_dict=None,
            producer_op_list=None
        )
        checkpoint_path = tf.train.latest_checkpoint(ckpt_path)
        print('pb file is loaded :', checkpoint_path)

        return graph


def save_model_setting(model_class):
    model_setting = vars(model_class)


class Reporter(object):
    def __init__(self, project_dir, subject_dir, train_setting, aug_list, model_params):
        self.save_path, self.train_log, self.start_time = self._report_ready_(project_dir, subject_dir,
                                                                              train_setting, aug_list, model_params)
        self.current_time = datetime.now()

    def _report_ready_(self, project_dir, subject_dir, train_setting, aug_list, model_params):
        save_dir = os.path.join(project_dir, subject_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        def define_save_path(lab_n, train_setting):
            project_title = train_setting['subject_title']
            train_date = datetime.now().strftime("%m.%d")
            report_name = '[%s-lab.%02d]_%s' % (train_date, lab_n, project_title)
            save_path = os.path.join(save_dir, report_name)
            return save_path, report_name

        def delete_log(save_path, report_name, log_len=10):
            # Check how much the last log worked and decide whether to delete the log or not
            pd_check_path = save_path + report_name + '_log.csv'
            if os.path.exists(pd_check_path):
                check_df = pd.read_csv(pd_check_path)
                if len(check_df) < log_len:
                    shutil.rmtree(save_path)
                    print("The log file is deleted by short log:", report_name)
                    return False
                else:
                    return True
            else:
                print(pd_check_path)
                shutil.rmtree(save_path)
                print("The log file is deleted by unfilled log:", report_name)
                return False

        lab_n = 0
        save_path, report_name = define_save_path(lab_n, train_setting)

        while os.path.exists(save_path):
            if delete_log(save_path, report_name):
                lab_n += 1
            save_path, report_name = define_save_path(lab_n, train_setting)

        os.makedirs(os.path.join(save_path, 'image'))
        os.makedirs(os.path.join(save_path, 'tboard'))
        print("Folder Created at :", save_path)

        report_path = os.path.join(save_path, 'train_parameters.txt')
        with open(report_path, 'w') as save_file:
            # Writing Train Setting
            save_file.write("[train_setting]")
            for key, value in train_setting.items():
                save_file.write("\n%s = %s" % (key, value))
            # Writing Data Augmentation List
            save_file.write("\n\n[data_augmentation]")
            for key, value in aug_list.items():
                save_file.write("\n%s = %s" % (key, value))
            # Writing Model Setting
            save_file.write("\n\n[model_setting]")
            for key, value in model_params.items():
                save_file.write("\n%s = %s" % (key, value))

        train_log = report_name + '_log.csv'
        start_time = datetime.now()
        return save_path, train_log, start_time

    def report_state(self, state, glob_n, epoch, num_epoch, batch_total, batch_n):
        if not os.path.exists(self.save_path + self.train_log):
            df = pd.DataFrame(columns=state.keys())
        else:
            df = pd.read_csv(self.save_path + self.train_log, index_col=0)

        cur_time = datetime.now()
        during_time = str(cur_time - self.start_time).split('.')[0]
        state = {key: '%.04f' % value for key, value in state.items()}

        state_num = 'step.%d' % (glob_n)
        len_format = '{:0' + str(len(str(num_epoch))) + '}'

        spd = (cur_time - self.current_time).seconds  # second per display
        self.current_time = cur_time

        # state_msg = '[' + len_format.format(num_epoch + 1) + '/' + len_format.format(epoch) + '] step.%d/%d' \
        #             % (batch_n + 1, batch_total) + " - " + str(state)[1:-1].replace("'", "") + \
        #             ' => Glob_Step:' + str(glob_n) + ', Time: ' + str(during_time) + ' (%dsec)' % spd

        state_msg = 'step.%d/%d' % (batch_n + 1, batch_total) + ' [' + len_format.format(num_epoch + 1) + '/' + \
                    len_format.format(epoch) + '] ' + str(state)[1:-1].replace("'", "") + \
                    ' => Glob_Step:' + str(glob_n + 1) + ', Time: ' + str(during_time) + ' (%dsec)' % spd

        progress_bar(batch_total, batch_n, state_msg)

        # state_num is removed from index
        new_row = pd.DataFrame(data=state, index=[num_epoch], columns=state.keys())
        new_df = df.append(new_row)
        new_df.to_csv(self.save_path + self.train_log)


def plot_report(log_list, subject_title, compare_col='cost'):
    df_list = []
    for log_file in log_list:
        subject_title, log_name = log_file.split('\\')[1].split("_", 1)[-1].split('_', 1)
        df = pd.read_csv(log_file, index_col=[0])
        df = df.rename(columns={compare_col: log_name})[log_name]
        df_list.append(df)

    report = pd.concat(df_list, axis=1)  # concatenate all DataFrames
    report.name = subject_title  # name subject title to DataFrame
    report[report.isnull()] = 0  # replace 'NaN' to 0

    # rename and sort index of DataFrame
    for n, idx in enumerate(report.index):
        step, idx_n = idx.split('.')
        report = report.rename(index={idx: int(idx_n)})
    report.sort_index(inplace=True)

    # Plot Graph for DataFrame of Report
    # https://pandas.pydata.org/pandas-docs/stable/visualization.html
    x_ticks = linspace(report.index[0], report.index[-1], 10)
    report.plot(colormap='jet', figsize=(20, 10), title='%s - %s' % (subject_title, compare_col), fontsize=15, xticks=x_ticks)
from glob import glob

import JuneNet

org_data_dir = 'c://workspace/data/mnist_png/training'
image_list = glob(org_data_dir + '/*/*.png')
label_list = [data_path.split('\\')[-2] for data_path in image_list]
id_list = [data_path.split('training\\')[-1].split('.')[0].replace("\\", '_') for data_path in image_list]


JuneNet.data_manager.generate_dataset(project_title='mnist_tmp', id_list=id_list,
                                      image_list=image_list, label_list=label_list, name_classes=range(10))

catdog_tmp = JuneNet.nn(project_title='mnist_tmp', work_dir='c://workspace/', data_ext='tfrecords',
                        num_gpus=None, tboard_record=True)

catdog_tmp.set_optimizer(subject_title='temp', mode='train', learning_rate=0.001, decay_step=3, lr_decay_factor=1.,
                         num_epochs=20, batch_size=64, input_shape=None, inbound_shape=None, grid_n=1, valid_rate=0.1,
                         model_name='vgg', loss_name='ce', optimizer='adam')

catdog_tmp.train(report_per_epoch=100, display_per_epoch=2, valid_per_epoch=50, evals_per_epoch=4, verbosity=3)

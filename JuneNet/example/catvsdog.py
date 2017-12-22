from glob import glob
import os

import JuneNet

org_data_dir = 'c://workspace/data/example_data/dogcat/train/'
image_list = glob(os.path.join(org_data_dir, '*/*.jpg'))
fname_list = ['_'.join(data_path.split("\\", -2)[1:]).split('.')[0] for data_path in image_list]
label_list = [fname.split("_")[0] for fname in fname_list]


target_list = JuneNet.data_manager.target_labels
changed_list = []
for i, (fname, label) in enumerate(zip(fname_list, label_list)):
    if fname in target_list:
        if label == 'dogs':
            label_list[i] = 'cats'
        else:
            label_list[i] = 'dogs'
        changed_list.append((fname, label_list[i]))
print("Changed List :", changed_list)

JuneNet.data_manager.generate_dataset(project_title='cat_dog', id_list=fname_list,
                                      image_list=image_list, label_list=label_list,
                                      name_classes=['cats', 'dogs'])

catdog_tmp = JuneNet.nn(project_title='cat_dog', work_dir='c://workspace/', data_ext='tfrecords',
                        num_gpus=None, tboard_record=True)

catdog_tmp.set_optimizer(subject_title='temp', mode='train', learning_rate=0.001, decay_step=3, lr_decay_factor=1.,
                         num_epochs=20, batch_size=48, input_shape=None, inbound_shape=None, grid_n=1, valid_rate=0.1,
                         model_name='wideresnet', loss_name='ce', optimizer='adam')

catdog_tmp.train(report_per_epoch=100, display_per_epoch=2, valid_per_epoch=50, evals_per_epoch=4, verbosity=3)

import tensorflow as tf
import os
import shutil


def create_saved_model(ckpt_dirs, input_tensor_name, output_tensor_name, tag_constants=["serve"], output_dir="output"):
    """Create the Tensorflow API Doc suggested saved_model for serving tensorflow graph.

    :param iterable ckpt_dirs: iterable checkpoint directory
    :param str input_tensor_name: name of the input tensor without prefix
    :param str output_tensor_name: name of the output tensor without prefix
    :param str output_dir: model export directory
    :param iterable tag_constants: tag_constants for tagging model
    :returns: create saved_model to directory provided
    """

    export_dir = os.path.join(output_dir, "model")
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    for ckpt_dir, tag_constant in zip(ckpt_dirs, tag_constants):
        with tf.Session() as sess:
            saver = tf.train.Saver
            ckpt = tf.train.get_checkpoint_state(ckpt_dir)

            # Restore variables from checkpoint if possible
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                saver = tf.train.import_meta_graph(os.path.join(ckpt_dir, ckpt_name + '.meta'), clear_devices=True)
                saver.restore(sess, os.path.join(ckpt_dir, ckpt_name))
            else:
                raise Exception(ckpt, "Please provide the decent checkpoint directory")

            input_tensor = sess.graph.get_tensor_by_name(input_tensor_name + ":0")
            output_tensor = sess.graph.get_tensor_by_name(output_tensor_name + ":0")
            input_tensor_info = tf.saved_model.utils.build_tensor_info(input_tensor)
            output_tensor_info = tf.saved_model.utils.build_tensor_info(output_tensor)
            signature = \
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'MRI': input_tensor_info},
                    outputs={'prediction': output_tensor_info},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
            builder.add_meta_graph_and_variables(sess,
                                                 [tag_constant],
                                                 signature_def_map={'segmentation': signature},
                                                 assets_collection=None)
    print("Model Saved ", ckpt_dir)
    builder.save()


if __name__ == '__main__':
    # ckpt_dirs = ["C:/workspace/train_checkpoints/train_0326_32k/"]
    # input_tensor_name = "IteratorGetNext"
    # output_tensor_name = 'last_block/conv/Relu'  # "output/BiasAdd"  # last_block/conv/Relu
    # tag_constants = ['serve']
    # output_dir = "C://workspace/SegEngine/train_checkpoints/"
    # create_saved_model(ckpt_dirs, input_tensor_name, output_tensor_name, output_dir=output_dir)

    ckpt_dirs = ['C:/workspace/junet_project/face_segmentation/05.29 - test/checkpoint/']
    input_tensor_name = "model/image"
    output_tensor_name = 'model/output/BiasAdd'  # "output/BiasAdd"  # last_block/conv/Relu
    tag_constants = ['serve']
    output_dir = "C://workspace/SegEngine/train_checkpoints/"
    create_saved_model(ckpt_dirs, input_tensor_name, output_tensor_name, output_dir=output_dir)

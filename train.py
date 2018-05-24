# -*- coding: utf-8 -*-

import tf_to_dataset
import model
import tensorflow as tf
import tensorflow.contrib.slim as slim

flags = tf.app.flags
flags.DEFINE_string("dataset_dir", None, "the file containing the tfrecord file.")
flags.DEFINE_string("logdir", None, "the path to log directory.")
FLAGS = flags.FLAGS

dataset = tf_to_dataset.get_dataset(FLAGS.dataset_dir, 24999, 2, "./labels.txt")
data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
image, label = data_provider.get(["image", "label"])
inputs, labels = tf.train.batch([image, label], batch_size=64, allow_smaller_final_batch=True)
labels = slim.one_hot_encoding(labels, 2)

cls_model = model.Model(is_training=True, num_classes=2)
preprocessed_inputs = cls_model.preprocess(inputs)
prediction_dict = cls_model.predict(preprocessed_inputs)
loss_dict = cls_model.loss(prediction_dict, labels)
loss = loss_dict["loss"]
acc_dict = cls_model.accuracy(prediction_dict, labels)
acc = acc_dict["acc"]
tf.summary.scalar("loss", loss)
tf.summary.scalar("acc", acc)

optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9)
train_op = slim.learning.create_train_op(loss, optimizer, summarize_gradients=True)
slim.learning.train(train_op=train_op, logdir=FLAGS.logdir, save_summaries_secs=20, 
    save_interval_secs=120)
    
if __name__ == "__main__":
    tf.app.run()

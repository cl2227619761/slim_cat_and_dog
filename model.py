# -*- coding: utf-8 -*-

import tf_to_dataset
import tensorflow as tf
import tensorflow.contrib.slim as slim
from abc import ABCMeta
from abc import abstractmethod


class BaseModel(object):
    """Abstract base class for any model."""
    __metaclass__ = ABCMeta
    
    def __init__(self, num_classes):
        """Constructor.
        
        Args:
          num_classes: Number of classes.
        """
        self._num_classes = num_classes
        
    @property
    def num_classes(self):
        return self._num_classes
    
    @abstractmethod
    def preprocess(self, inputs):
        """Input preprocessing
        
        Args:
          inputs: A float32 tensor with shape [batch_size, height, width, num_channesl]
          representing a batch of images.
        
        Returns:
          preprocessed inputs.
        """
        pass
    
    @abstractmethod
    def predict(self, preprocessed_inputs):
        """predict prediction tensors from inputs tensor.
        outputs of this function can be passed to loss or postprocess functions.
        
        Args:
          preprocessed_inputs: A float32 tensor with shape [batch_size, height, width, num_channels]
          representing a batch of images.
        
        Returns:
          prediction_dict: A dictionary holding prediction tensors to be passed to 
          the loss or postprocess functions.
        """
        pass
        
    @abstractmethod
    def postprocess(self, prediction_dict, **params):
        """convert predicted output tensors to final forms.
        
        Args:
          prediction_dict: A dictionary holding prediction tensors.
          **params: Additional keyword arguments.
          
        Returns:
          A dictionary containing the postprocessed results.
        """
        pass
        
    @abstractmethod
    def loss(self, prediction_dict, groundtruth_lists):
        """Compute scalar loss tensors with respect to provided groundtruth.
        
        Args:
          prediction_dict: A dictionary containing prediction tensors.
          groundtruth_lists: A list of tensors holding groundtruth information.
        
        Returns:
          A dictionary mapping strings (loss names) to scalar tensors.
        """
        pass
        
    @abstractmethod
    def accuracy(self, prediction_dict, groundtruth_lists):
        pass
        

class Model(BaseModel):
    """A simple 2-classification CNN model definition."""
    
    def __init__(self, is_training, num_classes):
        super(Model, self).__init__(num_classes=num_classes)
        self._is_training = is_training
        
    def preprocess(self, inputs):
        preprocessed_inputs = tf.to_float(inputs)
        preprocessed_inputs = tf.subtract(preprocessed_inputs, 128.0)
        preprocessed_inputs = tf.div(preprocessed_inputs, 128.0)
        return preprocessed_inputs
    
    def predict(self, preprocessed_inputs):
        with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=tf.nn.relu):
            net = preprocessed_inputs
            net = slim.repeat(net, 2, slim.conv2d, 32, [3, 3], scope="conv1")
            net = slim.max_pool2d(net, [2, 2], scope="pool1")
            net = slim.repeat(net, 2, slim.conv2d, 64, [3, 3], scope="conv2")
            net = slim.max_pool2d(net, [2, 2], scope="pool2")
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope="conv3")
            net = slim.max_pool2d(net, [2, 2], scope="pool3")
            net = slim.flatten(net, scope="flatten")
            net = slim.dropout(net, keep_prob=0.5, is_training=self._is_training)
            net = slim.fully_connected(net, 512, scope="fc1")
            net = slim.fully_connected(net, self.num_classes, activation_fn=None, scope="fc2")
        
        prediction_dict = {"logits": net}
        return prediction_dict
        
    def postprocess(self, prediction_dict):
        logits = prediction_dict["logits"]
        logits = tf.nn.softmax(logits)
        return logits
        
    def loss(self, prediction_dict, groundtruth_lists):
        logits = prediction_dict["logits"]
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=groundtruth_lists))
        loss_dict = {"loss": loss}
        return loss_dict
    
    def accuracy(self, prediction_dict, groundtruth_lists):
        logits = prediction_dict["logits"]
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(groundtruth_lists, 1))
        acc = tf.reduce_mean(tf.cast(correct_pred, "float"))
        acc_dict = {"acc": acc}
        return acc_dict
        

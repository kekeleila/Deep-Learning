# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains a collection of models which operate on variable-length sequences.
"""
import math

import model_utils as utils
import models
import tensorflow as tf
import tensorflow.contrib.slim as slim
import video_level_models
from PIL import Image
from tensorflow import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer("iterations", 30,
                     "Number of frames per batch for DBoF.")
flags.DEFINE_bool("dbof_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")
flags.DEFINE_bool(
    "sample_random_frames", True,
    "If true samples random frames (for frame level models). If false, a random"
    "sequence of frames is sampled instead.")
flags.DEFINE_integer("dbof_cluster_size", 8192,
                     "Number of units in the DBoF cluster layer.")
flags.DEFINE_integer("dbof_hidden_size", 1024,
                     "Number of units in the DBoF hidden layer.")
flags.DEFINE_string("dbof_pooling_method", "max",
                    "The pooling method used in the DBoF cluster layer. "
                    "Choices are 'average' and 'max'.")
flags.DEFINE_string("video_level_classifier_model", "MoeModel",
                    "Some Frame-Level models can be decomposed into a "
                    "generalized pooling operation followed by a "
                    "classifier layer")
flags.DEFINE_integer("lstm_cells", 1024, "Number of LSTM cells.")
flags.DEFINE_integer("lstm_layers", 2, "Number of LSTM layers.")


def create_1Dlayers(layer_input):
    conv1 = tf.layers.conv1d(layer_input, kernel_size=11, filters=96, strides=3, padding='SAME',
                             activation=tf.nn.relu)
    norm1 = tf.nn.local_response_normalization(tf.expand_dims(conv1, 3), 5, 2, 10e-4, 0.5)
    norm1 = tf.squeeze(norm1, 3)
    pool1 = tf.layers.max_pooling1d(norm1, pool_size=2, strides=2)
    conv2 = tf.layers.conv1d(pool1, kernel_size=5, filters=256, strides=1, padding='SAME', activation=tf.nn.relu)
    norm2 = tf.nn.local_response_normalization(tf.expand_dims(conv2, 3), 5, 2, 10e-4, 0.5)
    norm2 = tf.squeeze(norm2, 3)
    pool2 = tf.layers.max_pooling1d(norm2, pool_size=2, strides=2)
    conv3 = tf.layers.conv1d(pool2, kernel_size=3, filters=384, strides=1, padding='SAME', activation=tf.nn.relu)
    conv4 = tf.layers.conv1d(conv3, kernel_size=3, filters=384, strides=1, padding='SAME', activation=tf.nn.relu)
    conv5 = tf.layers.conv1d(conv4, kernel_size=3, filters=256, strides=1, padding='SAME', activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling1d(conv5, pool_size=2, strides=2)
    return pool3


def create_2Dlayers(layer_input):
    conv1 = tf.layers.conv2d(layer_input, kernel_size=11, filters=96, strides=(3, 3), padding='SAME',
                             activation=tf.nn.relu)
    norm1 = tf.nn.local_response_normalization(conv1, 5, 2, 10e-4, 0.5)
    pool1 = tf.layers.max_pooling2d(norm1, pool_size=2, strides=(2, 2))
    conv2 = tf.layers.conv2d(pool1, kernel_size=5, filters=256, strides=(1, 1), padding='SAME', activation=tf.nn.relu)
    norm2 = tf.nn.local_response_normalization(conv2, 5, 2, 10e-4, 0.5)
    pool2 = tf.layers.max_pooling2d(norm2, pool_size=2, strides=(2, 2))
    conv3 = tf.layers.conv2d(pool2, kernel_size=3, filters=384, strides=(1, 1), padding='SAME', activation=tf.nn.relu)
    conv4 = tf.layers.conv2d(conv3, kernel_size=3, filters=384, strides=(1, 1), padding='SAME', activation=tf.nn.relu)
    conv5 = tf.layers.conv2d(conv4, kernel_size=3, filters=256, strides=(1, 1), padding='SAME', activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(conv5, pool_size=2, strides=(2, 2))
    return pool3


def compute_output(last_pooling_layer, vocab_size):
    fc1 = tf.layers.dense(last_pooling_layer, 4096, activation=tf.nn.relu)
    fc2 = tf.layers.dense(fc1, 4096, activation=tf.nn.relu)
    output = slim.fully_connected(fc2, vocab_size, activation_fn=tf.nn.sigmoid,
                                  weights_regularizer=slim.l2_regularizer(1e-8))
    return output


class FrameLevelNNModelSingleFrame(models.BaseModel):
    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
        model_input = tf.transpose(tf.slice(model_input, [0, 0, 0], [1, 1, 1024]), perm=[0, 2, 1])
        pool3 = create_1Dlayers(model_input)
        output = compute_output(pool3, vocab_size)
        return {"predictions": output}


class FrameLevelNNModelSingleFrame2D(models.BaseModel):
    def create_model(self, model_input, vocab_size, num_frames, **unused_params):

        model_input = tf.reshape(tf.transpose(tf.slice(model_input, [0, 0, 0], [1, 1, 1024]), perm=[0, 2, 1]),
                                 [1, 32, 32, 1])
        pool3 = create_2Dlayers(model_input)
        output = compute_output(pool3, vocab_size)
        return {"predictions": output}

class FrameLevelRNN(models.BaseModel):
    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
        rnn_input = []
        for i in range(40):
            frame = tf.transpose(tf.slice(model_input, [0, i, 0], [1, 1, 1024]), perm=[0, 2, 1])
            pool = create_1Dlayers(frame)
            rnn_input.append(pool)
        rnn_input = tf.squeeze(rnn_input,1)
        nodes = tf.contrib.rnn.BasicLSTMCell(384)
        rnn_input = tf.unstack(rnn_input)
        output, final_state = tf.contrib.rnn.static_rnn(cell=nodes,dtype=tf.float32,
                           inputs=rnn_input)
        output = tf.stack(output)
        output = output[:,-1,:]
        output = tf.layers.dense(output, vocab_size, activation=tf.nn.relu)
        return {"predictions": output}




class FrameLevelNNModelEarlyFusion(models.BaseModel):
    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
        model_input = tf.transpose(tf.slice(model_input, [0, 0, 0], [1, 10, 1024]), perm=[0, 2, 1])
        pool3 = create_1Dlayers(model_input)
        output = compute_output(pool3, vocab_size)
        return {"predictions": output}


class FrameLevelNNModelLateFusion(models.BaseModel):
    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
        start_frame = tf.transpose(tf.slice(model_input, [0, 0, 0], [1, 1, 1024]), perm=[0, 2, 1])
        late_frame = tf.transpose(tf.slice(model_input, [0, 14, 0], [1, 1, 1024]), perm=[0, 2, 1])
        start_frame_output = create_1Dlayers(start_frame)
        late_frame_output = create_1Dlayers(late_frame)
        fc_input = tf.concat([start_frame_output, late_frame_output], 1)
        output = compute_output(fc_input, vocab_size)

        return {"predictions": output}


class FrameLevelNNModelSlowFusion(models.BaseModel):
    def layerBulk(self, layer_input):
        conv1 = tf.layers.conv1d(layer_input, kernel_size=11, filters=96, strides=3, padding='SAME',
                                 activation=tf.nn.relu)
        norm1 = tf.nn.local_response_normalization(tf.expand_dims(conv1, 3), 5, 2, 10e-4, 0.5)
        norm1 = tf.squeeze(norm1, 3)
        pool1 = tf.layers.max_pooling1d(norm1, pool_size=2, strides=2)
        return pool1

    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
        bulks1Inputs = []
        bulks1Inputs.append(tf.transpose(tf.slice(model_input, [0, 0, 0], [1, 4, 1024]), perm=[0, 2, 1]))
        bulks1Inputs.append(tf.transpose(tf.slice(model_input, [0, 1, 0], [1, 4, 1024]), perm=[0, 2, 1]))
        bulks1Inputs.append(tf.transpose(tf.slice(model_input, [0, 2, 0], [1, 4, 1024]), perm=[0, 2, 1]))
        bulks1Inputs.append(tf.transpose(tf.slice(model_input, [0, 3, 0], [1, 4, 1024]), perm=[0, 2, 1]))
        bulks1 = []
        for input in bulks1Inputs:
            bulks1.append(self.layerBulk(input))
        bulks2Input1 = tf.concat([bulks1[0], bulks1[1]], 1)
        bulks2Input2 = tf.concat([bulks1[2], bulks1[3]], 1)
        bulks2Output1 = self.layerBulk(bulks2Input1)
        bulks2Output2 = self.layerBulk(bulks2Input2)
        bulks3Input = tf.concat([bulks2Output1, bulks2Output2], 1)
        conv3 = tf.layers.conv1d(bulks3Input, kernel_size=3, filters=384, strides=1, padding='SAME',
                                 activation=tf.nn.relu)
        conv4 = tf.layers.conv1d(conv3, kernel_size=3, filters=384, strides=1, padding='SAME', activation=tf.nn.relu)
        conv5 = tf.layers.conv1d(conv4, kernel_size=3, filters=256, strides=1, padding='SAME', activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling1d(conv5, pool_size=2, strides=2)
        output = compute_output(pool3, vocab_size)
        return {"predictions": output}


class FrameLevelNNModelTwoStreams(models.BaseModel):
    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
        model_input = tf.transpose(tf.slice(model_input, [0, 0, 0], [1, 1, 1024]), perm=[0, 2, 1])
        high_stream_input = tf.slice(model_input, [0, 255, 0], [1, 512, 1])
        low_stream_input = tf.strided_slice(model_input, [0, 0, 0], [1, 1023, 1], [1, 2, 1])
        high_stream_output = create_1Dlayers(high_stream_input)
        low_stream_output = create_1Dlayers(low_stream_input)
        fc_input = tf.concat([high_stream_output, low_stream_output], 1)
        output = compute_output(fc_input, vocab_size)
        return {"predictions": output}


class FrameLevelLogisticModel(models.BaseModel):
    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
        """Creates a model which uses a logistic classifier over the average of the
        frame-level features.

        This class is intended to be an example for implementors of frame level
        models. If you want to train a model over averaged features it is more
        efficient to average them beforehand rather than on the fly.

        Args:
          model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                       input features.
          vocab_size: The number of classes in the dataset.
          num_frames: A vector of length 'batch' which indicates the number of
               frames for each video (before padding).

        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          'batch_size' x 'num_classes'.
        """
        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        feature_size = model_input.get_shape().as_list()[2]

        denominators = tf.reshape(
            tf.tile(num_frames, [1, feature_size]), [-1, feature_size])
        avg_pooled = tf.reduce_sum(model_input,
                                   axis=[1]) / denominators

        output = slim.fully_connected(
            avg_pooled, vocab_size, activation_fn=tf.nn.sigmoid,
            weights_regularizer=slim.l2_regularizer(1e-8))
        return {"predictions": output}


class DbofModel(models.BaseModel):
    """Creates a Deep Bag of Frames model.

    The model projects the features for each frame into a higher dimensional
    'clustering' space, pools across frames in that space, and then
    uses a configurable video-level model to classify the now aggregated features.

    The model will randomly sample either frames or sequences of frames during
    training to speed up convergence.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """

    def create_model(self,
                     model_input,
                     vocab_size,
                     num_frames,
                     iterations=None,
                     add_batch_norm=None,
                     sample_random_frames=None,
                     cluster_size=None,
                     hidden_size=None,
                     is_training=True,
                     **unused_params):
        iterations = iterations or FLAGS.iterations
        add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm
        random_frames = sample_random_frames or FLAGS.sample_random_frames
        cluster_size = cluster_size or FLAGS.dbof_cluster_size
        hidden1_size = hidden_size or FLAGS.dbof_hidden_size

        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        if random_frames:
            model_input = utils.SampleRandomFrames(model_input, num_frames,
                                                   iterations)
        else:
            model_input = utils.SampleRandomSequence(model_input, num_frames,
                                                     iterations)
        max_frames = model_input.get_shape().as_list()[1]
        feature_size = model_input.get_shape().as_list()[2]
        reshaped_input = tf.reshape(model_input, [-1, feature_size])
        tf.summary.histogram("input_hist", reshaped_input)

        if add_batch_norm:
            reshaped_input = slim.batch_norm(
                reshaped_input,
                center=True,
                scale=True,
                is_training=is_training,
                scope="input_bn")

        cluster_weights = tf.get_variable("cluster_weights",
                                          [feature_size, cluster_size],
                                          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))
        tf.summary.histogram("cluster_weights", cluster_weights)
        activation = tf.matmul(reshaped_input, cluster_weights)
        if add_batch_norm:
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=is_training,
                scope="cluster_bn")
        else:
            cluster_biases = tf.get_variable("cluster_biases",
                                             [cluster_size],
                                             initializer=tf.random_normal(stddev=1 / math.sqrt(feature_size)))
            tf.summary.histogram("cluster_biases", cluster_biases)
            activation += cluster_biases
        activation = tf.nn.relu6(activation)
        tf.summary.histogram("cluster_output", activation)

        activation = tf.reshape(activation, [-1, max_frames, cluster_size])
        activation = utils.FramePooling(activation, FLAGS.dbof_pooling_method)

        hidden1_weights = tf.get_variable("hidden1_weights",
                                          [cluster_size, hidden1_size],
                                          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
        tf.summary.histogram("hidden1_weights", hidden1_weights)
        activation = tf.matmul(activation, hidden1_weights)
        if add_batch_norm:
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=is_training,
                scope="hidden1_bn")
        else:
            hidden1_biases = tf.get_variable("hidden1_biases",
                                             [hidden1_size],
                                             initializer=tf.random_normal_initializer(stddev=0.01))
            tf.summary.histogram("hidden1_biases", hidden1_biases)
            activation += hidden1_biases
        activation = tf.nn.relu6(activation)
        tf.summary.histogram("hidden1_output", activation)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        return aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            **unused_params)


class LstmModel(models.BaseModel):
    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
        """Creates a model which uses a stack of LSTMs to represent the video.

        Args:
          model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                       input features.
          vocab_size: The number of classes in the dataset.
          num_frames: A vector of length 'batch' which indicates the number of
               frames for each video (before padding).

        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          'batch_size' x 'num_classes'.
        """
        lstm_size = FLAGS.lstm_cells
        number_of_layers = FLAGS.lstm_layers

        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0)
                for _ in range(number_of_layers)
            ])

        loss = 0.0

        outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                           sequence_length=num_frames,
                                           dtype=tf.float32)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)

        return aggregated_model().create_model(
            model_input=state[-1].h,
            vocab_size=vocab_size,
            **unused_params)

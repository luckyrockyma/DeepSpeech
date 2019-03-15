# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import numpy as np
import os
import pandas
import tensorflow as tf

from functools import partial
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from util.config import Config
from util.text import text_to_char_array


def read_csvs(csv_files):
    source_data = None
    for csv in csv_files:
        file = pandas.read_csv(csv, encoding='utf-8', na_filter=False)
        #FIXME: not cross-platform
        csv_dir = os.path.dirname(os.path.abspath(csv))
        file['wav_filename'] = file['wav_filename'].str.replace(r'(^[^/])', lambda m: os.path.join(csv_dir, m.group(1)))
        if source_data is None:
            source_data = file
        else:
            source_data = source_data.append(file)
    return source_data


def samples_to_mfccs(samples, sample_rate):
    spectrogram = contrib_audio.audio_spectrogram(samples, window_size=512, stride=320, magnitude_squared=True)
    mfccs = contrib_audio.mfcc(spectrogram, sample_rate, dct_coefficient_count=Config.n_input)
    mfccs = tf.reshape(mfccs, [-1, Config.n_input])

    return mfccs, tf.shape(mfccs)[0]


def file_to_features(wav_filename, transcript=None):
    samples = tf.read_file(wav_filename)
    decoded = contrib_audio.decode_wav(samples, desired_channels=1)
    features, features_len = samples_to_mfccs(decoded.audio, decoded.sample_rate)

    return features, features_len, transcript


def sparse_tuple_from(sequences, dtype=np.int32):
    r"""Creates a sparse representention of ``sequences``.
    Args:
        * sequences: a list of lists of type dtype where each element is a sequence
    Returns a tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), indices.max(0)[1]+1], dtype=np.int64)

    return tf.SparseTensor(indices=indices, values=values, dense_shape=shape)


def create_dataset(csvs, batch_size, cache_path):
    df = read_csvs(csvs)
    df.sort_values(by='wav_filesize', inplace=True)

    num_batches = len(df) // batch_size

    # Convert to character index arrays and then into a single SparseTensor
    transcripts = df['transcript'].apply(partial(text_to_char_array, alphabet=Config.alphabet))
    transcripts = sparse_tuple_from(transcripts.values)

    num_gpus = len(Config.available_devices)

    filenames = tf.data.Dataset.from_tensor_slices(df['wav_filename'].values)
    transcripts = tf.data.Dataset.from_tensor_slices(transcripts)

    def batch_fn(features, features_len, transcripts):
        features = tf.data.Dataset.zip((features, features_len))
        features = features.padded_batch(batch_size,
                                         padded_shapes=([None, Config.n_input], []))
        transcripts = transcripts.batch(batch_size)
        return tf.data.Dataset.zip((features, transcripts))

    dataset = (tf.data.Dataset.zip((filenames, transcripts))
                              .map(file_to_features, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                              .cache(cache_path)
                              .window(batch_size, drop_remainder=True).flat_map(batch_fn)
                              .prefetch(num_gpus)
                              .repeat())

    return dataset, num_batches

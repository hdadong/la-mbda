import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec
from tensorflow_probability import stats as tfps
from tf_agents.replay_buffers import episodic_replay_buffer

import la_mbda.utils as utils


class EpisodeBuffer(object):
    def __init__(self, safety):
        self._current_episode = {'observation1': [],
                                 'observation2': [],
                                 'action': [],
                                 'reward': [],
                                 'terminal': [],
                                 'info': []}
        if safety:
            self._current_episode['cost'] = []

    def store(self, transition):
        if len(self._current_episode['observation1']) == 0:
            for k, v in self._current_episode.items():
                if k == 'cost':
                    v.append(transition['info']['cost'])
                else:
                    v.append(transition[k])
            self._current_episode['observation1'].append(transition['next_observation1'])
            self._current_episode['observation2'].append(transition['next_observation2'])

        else:
            for k, v in self._current_episode.items():
                if k == 'observation1':
                    v.append(transition['next_observation1'])
                elif k == 'observation2':
                    v.append(transition['next_observation2'])
                elif k == 'cost':
                    v.append(transition['info']['cost'])
                else:
                    v.append(transition[k])

    def flush(self):
        episode_data = {k: np.array(v) for k, v in self._current_episode.items()
                        if k != 'info'}
        for v in self._current_episode.values():
            v.clear()
        return episode_data


class ReplayBuffer(tf.Module):
    def __init__(self, safety, observation_type, observation_shape, action_shape,
                 sequence_length, batch_size, seed, capacity=1000):
        super(ReplayBuffer, self).__init__()
        self._dtype = prec.global_policy().compute_dtype
        self._batch_size = batch_size
        self._sequence_length = sequence_length
        self._observation_type = observation_type
        self.observation_mean1 = tf.Variable(tf.zeros(observation_shape),
                                            dtype=np.float32, trainable=False)
        self.observation_variance1 = tf.Variable(tf.zeros(observation_shape),
                                                dtype=np.float32, trainable=False)
        self.observation_mean2 = tf.Variable(tf.zeros(observation_shape),
                                            dtype=np.float32, trainable=False)
        self.observation_variance2 = tf.Variable(tf.zeros(observation_shape),
                                                dtype=np.float32, trainable=False)
        self.running_episode_count = tf.Variable(0, trainable=False)
        self._current_episode = EpisodeBuffer(safety)
        self._safety = safety
        obs_dtype = tf.uint8 if observation_type in ['rgb_image', 'binary_image'] \
            else tf.float32
        data_spec = {'observation1': tf.TensorSpec(observation_shape, obs_dtype),
                     'observation2': tf.TensorSpec(observation_shape, obs_dtype),
                     'action': tf.TensorSpec(action_shape, self._dtype),
                     'reward': tf.TensorSpec((), self._dtype),
                     'terminal': tf.TensorSpec((), self._dtype)}
        if self._safety:
            data_spec['cost'] = tf.TensorSpec((), self._dtype)
        self._buffer = episodic_replay_buffer.EpisodicReplayBuffer(
            data_spec,
            seed=seed,
            capacity=capacity,
            buffer_size=1,
            dataset_drop_remainder=True,
            completed_only=True,
            begin_episode_fn=lambda _: True,
            end_episode_fn=lambda _: True)
        self._dataset = self._buffer.as_dataset(self._batch_size,
                                                self._sequence_length)
        self._dataset = self._dataset.map(self._preprocess,
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self._dataset = self._dataset.prefetch(10)

    @property
    def episode_count(self):
        return self.running_episode_count.value()

    def _finalize_episode(self):
        episode_data = self._current_episode.flush()
        if self._observation_type == 'dense':
            self._update_statistics(episode_data['observation1'], episode_data['observation2'])
        elif self._observation_type in ['rgb_image', 'binary_image']:
            bias = dict(rgb_image=0.5, binary_image=0.0).get(self._observation_type)
            episode_data['observation1'] = (episode_data['observation1'] + bias) * 255.0
            episode_data['observation1'].astype(np.uint8)

            episode_data['observation2'] = (episode_data['observation2'] + bias) * 255.0
            episode_data['observation2'].astype(np.uint8)

            self.running_episode_count.assign_add(1)

        self._buffer.add_sequence(episode_data, tf.cast(self.episode_count, tf.int64))

    def _update_statistics(self, observation1, observation2):
        tfps.assign_moving_mean_variance(
            observation1,
            self.observation_mean1,
            self.observation_variance1,
            axis=0,
            zero_debias_count=self.running_episode_count)

        tfps.assign_moving_mean_variance(
            observation2,
            self.observation_mean2,
            self.observation_variance2,
            axis=0,
            zero_debias_count=self.running_episode_count)
    def _preprocess(self, episode, _):
        if self._observation_type == 'dense':
            episode['observation1'] = utils.normalize_clip(
                tf.cast(episode['observation1'], tf.float32),
                tf.convert_to_tensor(self.observation_mean1),
                tf.sqrt(tf.convert_to_tensor(self.observation_variance1)), 10.0)
            episode['observation2'] = utils.normalize_clip(
                tf.cast(episode['observation2'], tf.float32),
                tf.convert_to_tensor(self.observation_mean2),
                tf.sqrt(tf.convert_to_tensor(self.observation_variance2)), 10.0)
        elif self._observation_type in ['rgb_image', 'binary_image']:
            bias = dict(rgb_image=0.5, binary_image=0.0).get(self._observation_type)
            episode['observation1'] = utils.preprocess(
                tf.cast(episode['observation1'], tf.float32), bias)

            bias = dict(rgb_image=0.5, binary_image=0.0).get(self._observation_type)
            episode['observation2'] = utils.preprocess(
                tf.cast(episode['observation2'], tf.float32), bias)
        else:
            raise RuntimeError("Invalid observation type")
        episode['observation1'] = tf.cast(episode['observation1'], self._dtype)
        episode['observation2'] = tf.cast(episode['observation2'], self._dtype)

        episode['terminal'] = episode['terminal'][:, :-1]
        episode['reward'] = episode['reward'][:, :-1]
        episode['action'] = episode['action'][:, :-1]
        if self._safety:
            episode['cost'] = episode['cost'][:, :-1]
        return episode

    def store(self, transition):
        self._current_episode.store(transition)
        if transition['terminal'] or transition['info'].get('TimeLimit.truncated'):
            self._finalize_episode()

    def sample(self, n_batches):
        return self._dataset.take(n_batches)

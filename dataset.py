import os
from random import shuffle
import random
import numpy as np
import tensorflow as tf
# from tensorflow.data import Dataset, Iterator
from tensorflow.python.ops import control_flow_ops
data = tf.data
Dataset = data.Dataset
Iterator = data.Iterator

SEED = 42
random.seed(SEED)

import pandas as pd

class ImageDataset():
    def __init__(self, image_path, metadata_path, mode, image_size, batch_size, output_path, selected_attrs, augmentation):
      self.image_path = image_path
      self.image_path = image_path
      self.image_size = image_size
      self.batch_size = batch_size
      self.ouput_path = output_path
      tf.gfile.MakeDirs(output_path)
      self.selected_attrs = selected_attrs
      self.augmentation = augmentation
      # self.selected_attrs = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs',
      #                        'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
      #                        'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
      #                        'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes',
      #                        'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
      #                        'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
      #                        'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
      self.mode = mode
      self.lines = open(metadata_path, 'r').readlines()
      self.num_data = int(self.lines[0])
      self.attr2idx = {}
      self.idx2attr = {}

      print('Start preprocessing dataset..!')
      if os.path.exists(os.path.join(output_path, 'train_filenames.npy')):
        self.train_filenames = np.load(os.path.join(output_path, 'train_filenames.npy'))
        self.train_labels = np.load(os.path.join(output_path, 'train_labels.npy'))
        self.test_filenames = np.load(os.path.join(output_path, 'test_filenames.npy'))
        self.test_labels = np.load(os.path.join(output_path, 'test_labels.npy'))
        self.test_images = np.load(os.path.join(output_path, 'test_images.npy'))
        self.test_fake_labels = np.load(os.path.join(output_path, 'test_fake_labels.npy'))
      else:
        self.preprocess()
        np.save(os.path.join(output_path, 'train_filenames.npy'), self.train_filenames)
        np.save(os.path.join(output_path, 'test_images.npy'), self.test_images)
        np.save(os.path.join(output_path, 'train_labels.npy'), self.train_labels)
        np.save(os.path.join(output_path, 'test_filenames.npy'), self.test_filenames)
        np.save(os.path.join(output_path, 'test_labels.npy'), self.test_labels)
        np.save(os.path.join(output_path, 'test_fake_labels.npy'), self.test_fake_labels)
      print('Finished preprocessing dataset..!')

      if self.mode == 'train':
        self.num_data = len(self.train_filenames)
      elif self.mode == 'test':
        self.num_data = len(self.test_filenames)

      print(self.num_data)
      self.train_data = Dataset.from_tensor_slices((self.train_filenames, self.train_labels))
      self.train_data = self.train_data.shuffle(self.num_data).repeat()
      self.train_data = self.train_data.batch(batch_size)

      self.train_data = self.train_data.map(self.input_parser_train)
      self.train_iterator = Iterator.from_structure(self.train_data.output_types,
                                                    self.train_data.output_shapes)
      self.train_next = self.train_iterator.get_next()
      self.train_init = self.train_iterator.make_initializer(self.train_data)

      self.test_data = Dataset.from_tensor_slices(
        (self.test_filenames, self.test_images, self.test_labels, self.test_fake_labels))
      if self.mode == 'train':
        self.test_data = self.test_data.repeat().batch(batch_size)
      else:
        self.test_data = self.test_data.batch(batch_size)
      self.test_data = self.test_data.map(self.input_parser_test)
      self.test_iterator = Iterator.from_structure(self.test_data.output_types,
                                                   self.test_data.output_shapes)
      self.test_next = self.test_iterator.get_next()
      self.test_init = self.test_iterator.make_initializer(self.test_data)

    def custom_resize(self, image):
      initial_width = tf.shape(image)[0]
      initial_height = tf.shape(image)[1]

      def _resize(x, y):
        # Take the greater value, and use it for the ratio
        max_ = tf.minimum(initial_height, initial_width)
        # argmax_ = tf.argmax([initial_height, initial_width])
        ratio = tf.to_float(max_) / tf.constant(self.image_size[1], dtype=tf.float32)

        new_width = tf.to_float(initial_width) / ratio
        new_height = tf.to_float(initial_height) / ratio

        return tf.to_int32(new_width), tf.to_int32(new_height)

      def _useless(x, y):
        return x, y

      new_w, new_h = tf.cond(tf.logical_or(
        tf.greater(initial_width, tf.constant(self.image_size[1])),
        tf.greater(initial_height, tf.constant(self.image_size[0]))
      ),
        lambda: _resize(initial_width, initial_height),
        lambda: _useless(initial_width, initial_height))

      resized_image = tf.image.resize_images(image, [new_w, new_h])
      resized_image = tf.cast(resized_image, tf.uint8)
      return resized_image

    def distort_color(self, image, color_ordering=0, fast_mode=True, scope=None):
      with tf.name_scope(scope, 'distort_color', [image]):
        # if fast_mode:
        #   if color_ordering == 0:
        #     image = tf.image.random_brightness(image, max_delta=32. / 255.)
        #     image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        #   else:
        #     image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        #     image = tf.image.random_brightness(image, max_delta=32. / 255.)
        # else:
        if color_ordering == 0:
          image = tf.image.random_brightness(image, max_delta=32. / 255.)
          # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
          # image = tf.image.random_hue(image, max_delta=0.2)
          image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
          # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
          image = tf.image.random_brightness(image, max_delta=32. / 255.)
          image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
          # image = tf.image.random_hue(image, max_delta=0.2)
        elif color_ordering == 2:
          image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
          # image = tf.image.random_hue(image, max_delta=0.2)
          image = tf.image.random_brightness(image, max_delta=32. / 255.)
          # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        elif color_ordering == 3:
          # image = tf.image.random_hue(image, max_delta=0.2)
          # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
          image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
          image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
          raise ValueError('color_ordering must be in [0, 3]')

        # The random_* ops do not necessarily clamp.
        return tf.clip_by_value(image, 0.0, 1.0)

    def apply_with_random_selector(self, x, func, num_cases):
      sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
      return control_flow_ops.merge([
        func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
        for case in range(num_cases)])[0]

    def input_parser_train(self, img_path, label):
      # convert the label to one-hot encoding
      # one_hot = tf.one_hot(label, len(self.selected_attrs))

      # read the img from file
      img = tf.map_fn(lambda x: tf.read_file(x), img_path)
      img_decoded = tf.map_fn(lambda x: tf.image.decode_jpeg(x, channels=3), img, dtype=tf.uint8)
      img_custom_resized = tf.map_fn(lambda x: self.custom_resize(x), img_decoded)
      img_resized = tf.map_fn(
        lambda x: tf.image.resize_image_with_crop_or_pad(x, self.image_size[0], self.image_size[1]),
        img_custom_resized)
      img_float = tf.map_fn(lambda x: tf.to_float(x), img_resized, dtype=tf.float32)
      img_normalized = tf.map_fn(lambda x: (1.0 / 255.0) * x, img_float)
      if self.augmentation:
        img_normalized = tf.map_fn(lambda x: self.apply_with_random_selector(
        x,
        lambda x, ordering: self.distort_color(x, ordering, False),
        num_cases=4), img_normalized)
      img_flipped = tf.map_fn(lambda x: tf.image.random_flip_left_right(x, SEED), img_normalized)
      img_final = tf.map_fn(lambda x: 2.0 * (x - 0.5), img_flipped)
      return img_final, label

    def input_parser_test(self, img_path, filename, label, fake_label):
      img = tf.map_fn(lambda x: tf.read_file(x), img_path)
      img_decoded = tf.map_fn(lambda x: tf.image.decode_jpeg(x, channels=3), img, dtype=tf.uint8)
      img_custom_resized = tf.map_fn(lambda x: self.custom_resize(x), img_decoded)
      img_resized = tf.map_fn(
        lambda x: tf.image.resize_image_with_crop_or_pad(x, self.image_size[0], self.image_size[1]),
        img_custom_resized)
      img_float = tf.map_fn(lambda x: tf.to_float(x), img_resized, dtype=tf.float32)
      img_normalized = tf.map_fn(lambda x: (2.0 / 255.0) * x - 1.0, img_float)
      return img_normalized, filename, label, fake_label

    def multi_label_list(self, label):
      y = [[1, 0, 0],  # black hair
           [0, 1, 0],  # blond hair
           [0, 0, 1]]  # brown hair
      label_list = []
      for i in range(4):
        fixed_label = label.copy()
        if i in [0, 1, 3]:  # Hair color to brown
          fixed_label[:3] = y[2]
        if i in [0, 2, 3]:  # Gender
          fixed_label[3] = 0 if fixed_label[3] == 1 else 1
        if i in [1, 2, 3]:  # Aged
          fixed_label[4] = 0 if fixed_label[4] == 1 else 1
        label_list.append(fixed_label)

      return label_list

    def single_label_list(self, label):
      #y = [[1, 0, 0],  # black hair
      #     [0, 1, 0],  # blond hair
      #     [0, 0, 1]]  # brown hair
      # choices = [c for c in range(len(self.selected_attrs))]# if c not in [8, 9, 11, 16]]
      # selected_choices = np.random.choice(choices, 10)
      label_list = []
      for i in range(len(self.selected_attrs)):
        fixed_label = label.copy()
        #iif i in [:
        #  fixed_label[:3] = y[i]
        #else:
        fixed_label[i] = 0 if fixed_label[i] == 1 else 1  # opposite value
        label_list.append(fixed_label)

      return label_list

    def preprocess(self):
      attrs = self.lines[1].split()
      for i, attr in enumerate(attrs):
        self.attr2idx[attr] = i
        self.idx2attr[i] = attr

      self.train_filenames = []
      self.train_labels = []
      self.test_filenames = []
      self.test_labels = []
      self.test_fake_labels = []
      self.test_images = []

      lines = self.lines[2:]
      random.shuffle(lines)  # random shuffling
      for i, line in enumerate(lines):

        splits = line.split()
        filename = splits[0]
        values = splits[1:]

        label = []
        for idx, value in enumerate(values):
          attr = self.idx2attr[idx]

          if attr in self.selected_attrs:
            if value == '1':
              label.append(1)
            else:
              label.append(0)

        if (i + 1) < 2000:
          single_fix_label_list = self.single_label_list(label)
          #multi_fix_label_list = self.multi_label_list(label)
          self.test_filenames.append(os.path.join(self.image_path, filename))
          self.test_labels.append(label)
          #self.test_fake_labels.append((single_fix_label_list + multi_fix_label_list))
          self.test_fake_labels.append(single_fix_label_list)
          self.test_images.append(filename)
        else:
          self.train_filenames.append(os.path.join(self.image_path, filename))
          self.train_labels.append(label)


class GenerationImageDataset():
  def __init__(self, image_path, metadata_path, mode, image_size, batch_size, output_path, selected_attrs,
               augmentation):
    self.image_path = image_path
    self.image_path = image_path
    self.image_size = image_size
    self.batch_size = batch_size
    self.ouput_path = output_path
    tf.gfile.MakeDirs(output_path)
    self.selected_attrs = selected_attrs
    self.augmentation = augmentation

    self.mode = mode
    self.lines = open(metadata_path, 'r').readlines()
    self.num_data = int(self.lines[0])
    self.attr2idx = {}
    self.idx2attr = {}

    print('Start preprocessing dataset..!')
    if os.path.exists(os.path.join(output_path, 'train_filenames.npy')):
      self.train_filenames = np.load(os.path.join(output_path, 'train_filenames.npy'))
      self.train_labels = np.load(os.path.join(output_path, 'train_labels.npy'))
      self.test_filenames = np.load(os.path.join(output_path, 'test_filenames.npy'))
      self.test_images = np.load(os.path.join(output_path, 'test_images.npy'))
      self.test_labels = np.load(os.path.join(output_path, 'test_labels.npy'))
      self.test_fake_labels = np.load(os.path.join(output_path, 'test_fake_labels.npy'))
    else:
      self.preprocess()
      self.test_fake_labels = []
      for filename, label in zip(self.test_filenames, self.test_labels):
        single_fix_label_list = self.single_label_list(label)
        self.test_fake_labels.append(single_fix_label_list)

      self.test_data = Dataset.from_tensor_slices((self.test_filenames, self.test_images, self.test_labels, self.test_fake_labels))
      np.save(os.path.join(output_path, 'train_filenames.npy'), self.train_filenames)
      np.save(os.path.join(output_path, 'test_images.npy'), self.test_images)
      np.save(os.path.join(output_path, 'train_labels.npy'), self.train_labels)
      np.save(os.path.join(output_path, 'test_filenames.npy'), self.test_filenames)
      np.save(os.path.join(output_path, 'test_labels.npy'), self.test_labels)
      np.save(os.path.join(output_path, 'test_fake_labels.npy'), self.test_fake_labels)
    print('Finished preprocessing dataset..!')

    if self.mode == 'train':
      self.num_data = len(self.train_filenames)
    elif self.mode == 'test':
      self.num_data = len(self.test_filenames)

    print(self.num_data)
    self.train_data = Dataset.from_tensor_slices((self.train_filenames, self.train_labels))
    self.train_data = self.train_data.repeat()
    self.train_data = self.train_data.batch(batch_size)

    self.train_data = self.train_data.map(self.input_parser_train)
    self.train_iterator = Iterator.from_structure(self.train_data.output_types,
                                                  self.train_data.output_shapes)
    self.train_next = self.train_iterator.get_next()
    self.train_init = self.train_iterator.make_initializer(self.train_data)

    self.test_data = Dataset.from_tensor_slices((self.test_filenames, self.test_images, self.test_labels, self.test_fake_labels))
    if self.mode == 'train':
      self.test_data = self.test_data.repeat().batch(batch_size)
    else:
      self.test_data = self.test_data.batch(batch_size)
    self.test_data = self.test_data.map(self.input_parser_test)
    self.test_iterator = Iterator.from_structure(self.test_data.output_types,
                                                 self.test_data.output_shapes)
    self.test_next = self.test_iterator.get_next()
    self.test_init = self.test_iterator.make_initializer(self.test_data)

  def custom_resize(self, image):
    initial_width = tf.shape(image)[0]
    initial_height = tf.shape(image)[1]

    def _resize(x, y):
      # Take the greater value, and use it for the ratio
      max_ = tf.minimum(initial_height, initial_width)
      # argmax_ = tf.argmax([initial_height, initial_width])
      ratio = tf.to_float(max_) / tf.constant(self.image_size[1], dtype=tf.float32)

      new_width = tf.to_float(initial_width) / ratio
      new_height = tf.to_float(initial_height) / ratio

      return tf.to_int32(new_width), tf.to_int32(new_height)

    def _useless(x, y):
      return x, y

    new_w, new_h = tf.cond(tf.logical_or(
      tf.greater(initial_width, tf.constant(self.image_size[1])),
      tf.greater(initial_height, tf.constant(self.image_size[0]))
    ),
      lambda: _resize(initial_width, initial_height),
      lambda: _useless(initial_width, initial_height))

    resized_image = tf.image.resize_images(image, [new_w, new_h])
    resized_image = tf.cast(resized_image, tf.uint8)
    return resized_image

  def distort_color(self, image, color_ordering=0, fast_mode=True, scope=None):
    with tf.name_scope(scope, 'distort_color', [image]):
      # if fast_mode:
      #   if color_ordering == 0:
      #     image = tf.image.random_brightness(image, max_delta=32. / 255.)
      #     image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      #   else:
      #     image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      #     image = tf.image.random_brightness(image, max_delta=32. / 255.)
      # else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        # image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      elif color_ordering == 1:
        # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        # image = tf.image.random_hue(image, max_delta=0.2)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        # image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      elif color_ordering == 3:
        # image = tf.image.random_hue(image, max_delta=0.2)
        # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

      # The random_* ops do not necessarily clamp.
      return tf.clip_by_value(image, 0.0, 1.0)

  def apply_with_random_selector(self, x, func, num_cases):
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    return control_flow_ops.merge([
      func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
      for case in range(num_cases)])[0]

  def input_parser_train(self, img_path, label):
    # convert the label to one-hot encoding
    # one_hot = tf.one_hot(label, len(self.selected_attrs))

    # read the img from file
    img = tf.map_fn(lambda x: tf.read_file(x), img_path)
    img_decoded = tf.map_fn(lambda x: tf.image.decode_jpeg(x, channels=3), img, dtype=tf.uint8)
    img_custom_resized = tf.map_fn(lambda x: self.custom_resize(x), img_decoded)
    img_resized = tf.map_fn(
      lambda x: tf.image.resize_image_with_crop_or_pad(x, self.image_size[0], self.image_size[1]),
      img_custom_resized)
    img_float = tf.map_fn(lambda x: tf.to_float(x), img_resized, dtype=tf.float32)
    img_normalized = tf.map_fn(lambda x: (1.0 / 255.0) * x, img_float)
    if self.augmentation:
      img_normalized = tf.map_fn(lambda x: self.apply_with_random_selector(
        x,
        lambda x, ordering: self.distort_color(x, ordering, False),
        num_cases=4), img_normalized)
    img_flipped = tf.map_fn(lambda x: tf.image.random_flip_left_right(x, SEED), img_normalized)
    img_final = tf.map_fn(lambda x: 2.0 * (x - 0.5), img_flipped)
    return img_final, label

  def input_parser_test(self, img_path, filename, label, fake_label):
    img = tf.map_fn(lambda x: tf.read_file(x), img_path)
    img_decoded = tf.map_fn(lambda x: tf.image.decode_jpeg(x, channels=3), img, dtype=tf.uint8)
    img_custom_resized = tf.map_fn(lambda x: self.custom_resize(x), img_decoded)
    img_resized = tf.map_fn(
      lambda x: tf.image.resize_image_with_crop_or_pad(x, self.image_size[0], self.image_size[1]),
      img_custom_resized)
    img_float = tf.map_fn(lambda x: tf.to_float(x), img_resized, dtype=tf.float32)
    img_normalized = tf.map_fn(lambda x: (2.0 / 255.0) * x - 1.0, img_float)
    return img_normalized, filename, label, fake_label

  def multi_label_list(self, label):
    y = [[1, 0, 0],  # black hair
         [0, 1, 0],  # blond hair
         [0, 0, 1]]  # brown hair
    label_list = []
    for i in range(4):
      fixed_label = label.copy()
      if i in [0, 1, 3]:  # Hair color to brown
        fixed_label[:3] = y[2]
      if i in [0, 2, 3]:  # Gender
        fixed_label[3] = 0 if fixed_label[3] == 1 else 1
      if i in [1, 2, 3]:  # Aged
        fixed_label[4] = 0 if fixed_label[4] == 1 else 1
      label_list.append(fixed_label)

    return label_list

  def single_label_list(self, label):
    # y = [[1, 0, 0],  # black hair
    #     [0, 1, 0],  # blond hair
    #     [0, 0, 1]]  # brown hair
    # choices = [c for c in range(len(self.selected_attrs))]# if c not in [8, 9, 11, 16]]
    # selected_choices = np.random.choice(choices, 10)
    label_list = []
    for i in range(len(self.selected_attrs)):
      fixed_label = label.copy()
      # iif i in [:
      #  fixed_label[:3] = y[i]
      # else:
      fixed_label[i] = 0 if fixed_label[i] == 1 else 1  # opposite value
      label_list.append(fixed_label)

    return label_list

  def preprocess(self):
    attrs = self.lines[1].split()
    for i, attr in enumerate(attrs):
      self.attr2idx[attr] = i
      self.idx2attr[i] = attr

    self.train_filenames = []
    self.test_images = []
    self.train_labels = []
    self.test_filenames = []
    self.test_labels = []
    self.test_fake_labels = []

    lines = self.lines[2:]
    random.shuffle(lines)  # random shuffling
    for i, line in enumerate(lines):
      splits = line.split()
      filename = splits[0]
      values = splits[1:]

      label = []
      for idx, value in enumerate(values):
        attr = self.idx2attr[idx]

        if attr in self.selected_attrs:
          if value == '1':
            label.append(1)
          else:
            label.append(0)

      if (i + 1) < 2000:
        single_fix_label_list = self.single_label_list(label)
        # multi_fix_label_list = self.multi_label_list(label)
        self.test_filenames.append(os.path.join(self.image_path, filename))
        self.test_labels.append(label)
        self.test_images.append(filename)
        # self.test_fake_labels.append((single_fix_label_list + multi_fix_label_list))
        self.test_fake_labels.append(single_fix_label_list)
      else:
        self.train_filenames.append(os.path.join(self.image_path, filename))
        self.train_labels.append(label)
        self.test_filenames.append(os.path.join(self.image_path, filename))
        self.test_images.append(filename)
        self.test_labels.append(label)


class ImageFolder():
  def __init__(self, image_path, image_size, batch_size, output_path, selected_attrs):
    self.image_path = os.path.join(image_path, 'images')

    # self.test_filenames = [os.path.join(self.image_path, f) for f in os.listdir(self.image_path)]
    # self.test_filenames.sort()
    self.test_filenames = []
    self.test_labels = []
    self.test_images = []
    self.test_fake_labels = []
    self.image_size = image_size
    self.labels_path = os.path.join(image_path, 'list_attrs.txt')
    self.selected_attrs = selected_attrs
    self.attr2idx = {}
    self.idx2attr = {}

    self.read_labels()

    self.train_filenames = np.load(os.path.join(output_path, 'train_filenames.npy'))
    self.train_labels = np.load(os.path.join(output_path, 'train_labels.npy'))
    self.train_data = Dataset.from_tensor_slices((self.train_filenames, self.train_labels))
    self.train_data = self.train_data.batch(batch_size)
    self.train_data = self.train_data.map(self.input_parser_train)
    self.train_iterator = Iterator.from_structure(self.train_data.output_types,
                                                  self.train_data.output_shapes)
    self.train_next = self.train_iterator.get_next()
    self.train_init = self.train_iterator.make_initializer(self.train_data)
    for filename, label in zip(self.test_filenames, self.test_labels):
      single_fix_label_list = self.single_label_list(label)
      self.test_fake_labels.append(single_fix_label_list)
      # print(self.test_fake_labels)

    self.test_data = Dataset.from_tensor_slices((self.test_filenames, self.test_images, self.test_labels, self.test_fake_labels))
    self.test_data = self.test_data.batch(batch_size)
    self.test_data = self.test_data.map(self.input_parser_test)
    self.test_iterator = Iterator.from_structure(self.test_data.output_types,
                                                 self.test_data.output_shapes)
    self.test_next = self.test_iterator.get_next()
    self.test_init = self.test_iterator.make_initializer(self.test_data)

  def read_labels(self):
    self.lines = open(self.labels_path, 'r').readlines()
    attrs = self.lines[0].split()
    for i, attr in enumerate(attrs):
      self.attr2idx[attr] = i
      self.idx2attr[i] = attr

    lines = self.lines[1:]
    for i, line in enumerate(lines):

      splits = line.split()
      filename = splits[0]
      values = splits[1:]

      label = []
      for idx, value in enumerate(values):
        attr = self.idx2attr[idx]

        if attr in self.selected_attrs:
          if value == '1':
            label.append(1)
          else:
            label.append(0)
      self.test_labels.append(label)
      self.test_filenames.append(os.path.join(self.image_path, filename))
      self.test_images.append(filename)

  def custom_resize(self, image):
    initial_width = tf.shape(image)[0]
    initial_height = tf.shape(image)[1]

    def _resize(x, y):
      # Take the greater value, and use it for the ratio
      max_ = tf.minimum(initial_height, initial_width)
      # argmax_ = tf.argmax([initial_height, initial_width])
      ratio = tf.to_float(max_) / tf.constant(self.image_size[1], dtype=tf.float32)

      new_width = tf.to_float(initial_width) / ratio
      new_height = tf.to_float(initial_height) / ratio

      return tf.to_int32(new_width), tf.to_int32(new_height)

    def _useless(x, y):
      return x, y

    new_w, new_h = tf.cond(tf.logical_or(
      tf.greater(initial_width, tf.constant(self.image_size[1])),
      tf.greater(initial_height, tf.constant(self.image_size[0]))
    ),
      lambda: _resize(initial_width, initial_height),
      lambda: _useless(initial_width, initial_height))

    resized_image = tf.image.resize_images(image, [new_w, new_h])
    resized_image = tf.cast(resized_image, tf.uint8)
    return resized_image

  def input_parser_train(self, img_path, label):
    # convert the label to one-hot encoding
    # one_hot = tf.one_hot(label, len(self.selected_attrs))

    # read the img from file
    img = tf.map_fn(lambda x: tf.read_file(x), img_path)
    img_decoded = tf.map_fn(lambda x: tf.image.decode_jpeg(x, channels=3), img, dtype=tf.uint8)
    img_custom_resized = tf.map_fn(lambda x: self.custom_resize(x), img_decoded)
    img_resized = tf.map_fn(
      lambda x: tf.image.resize_image_with_crop_or_pad(x, self.image_size[0], self.image_size[1]),
      img_custom_resized)
    img_float = tf.map_fn(lambda x: tf.to_float(x), img_resized, dtype=tf.float32)
    img_flipped = tf.map_fn(lambda x: tf.image.random_flip_left_right(x, SEED), img_float)
    img_normalized = tf.map_fn(lambda x: 2.0 * ((1.0 / 255.0) * x - 0.5), img_flipped)
    return img_normalized, label

  def input_parser_test(self, img_path, filename, label, fake_label):
    img = tf.map_fn(lambda x: tf.read_file(x), img_path)
    img_decoded = tf.map_fn(lambda x: tf.image.decode_jpeg(x, channels=3), img, dtype=tf.uint8)
    img_custom_resized = tf.map_fn(lambda x: self.custom_resize(x), img_decoded)
    img_resized = tf.map_fn(
      lambda x: tf.image.resize_image_with_crop_or_pad(x, self.image_size[0], self.image_size[1]),
      img_custom_resized)
    img_float = tf.map_fn(lambda x: tf.to_float(x), img_resized, dtype=tf.float32)
    img_normalized = tf.map_fn(lambda x: (2.0 / 255.0) * x - 1.0, img_float)
    return img_normalized, filename, label, fake_label

  def single_label_list(self, label):
    # y = [[1, 0, 0],  # black hair
    #     [0, 1, 0],  # blond hair
    #     [0, 0, 1]]  # brown hair
    # choices = [c for c in range(len(self.selected_attrs))]# if c not in [8, 9, 11, 16]]
    # selected_choices = np.random.choice(choices, 10)
    label_list = []
    for i in range(len(self.selected_attrs)):
      fixed_label = label.copy()
      # iif i in [:
      #  fixed_label[:3] = y[i]
      # else:
      fixed_label[i] = 0 if fixed_label[i] == 1 else 1  # opposite value
      label_list.append(fixed_label)

    return label_list

class TestImages():
  def __init__(self, image_path, image_size, batch_size, output_path, selected_attrs):
    self.image_path = image_path

    self.image_size = image_size
    self.selected_attrs = selected_attrs
    self.test_filenames = np.load(os.path.join(output_path, 'test_filenames.npy'))
    self.test_labels = np.load(os.path.join(output_path, 'test_labels.npy'))
    self.test_images = np.load(os.path.join(output_path, 'test_images.npy'))
    self.test_fake_labels = []

    for filename, label in zip(self.test_filenames, self.test_labels):
      single_fix_label_list = self.single_label_list(label)
      self.test_fake_labels.append(single_fix_label_list)

    self.test_data = Dataset.from_tensor_slices(
      (self.test_filenames, self.test_images, self.test_labels, np.asarray(self.test_fake_labels)))
    self.test_data = self.test_data.batch(batch_size)
    self.test_data = self.test_data.map(self.input_parser_test)
    self.test_iterator = Iterator.from_structure(self.test_data.output_types,
                                                 self.test_data.output_shapes)
    self.test_next = self.test_iterator.get_next()
    self.test_init = self.test_iterator.make_initializer(self.test_data)

    self.train_filenames = np.load(os.path.join(output_path, 'train_filenames.npy'))
    self.train_labels = np.load(os.path.join(output_path, 'train_labels.npy'))
    self.train_data = Dataset.from_tensor_slices((self.train_filenames, self.train_labels))
    self.train_data = self.train_data.batch(batch_size)
    self.train_data = self.train_data.map(self.input_parser_train)
    self.train_iterator = Iterator.from_structure(self.train_data.output_types,
                                                  self.train_data.output_shapes)
    self.train_next = self.train_iterator.get_next()
    self.train_init = self.train_iterator.make_initializer(self.train_data)

  def read_labels(self):
    self.lines = open(self.labels_path, 'r').readlines()
    attrs = self.lines[0].split()
    for i, attr in enumerate(attrs):
      self.attr2idx[attr] = i
      self.idx2attr[i] = attr

    lines = self.lines[1:]
    for i, line in enumerate(lines):

      splits = line.split()
      filename = splits[0]
      values = splits[1:]

      label = []
      for idx, value in enumerate(values):
        attr = self.idx2attr[idx]

        if attr in self.selected_attrs:
          if value == '1':
            label.append(1)
          else:
            label.append(0)
      self.test_labels.append(label)

  def custom_resize(self, image):
    initial_width = tf.shape(image)[0]
    initial_height = tf.shape(image)[1]

    def _resize(x, y):
      # Take the greater value, and use it for the ratio
      max_ = tf.minimum(initial_height, initial_width)
      # argmax_ = tf.argmax([initial_height, initial_width])
      ratio = tf.to_float(max_) / tf.constant(self.image_size[1], dtype=tf.float32)

      new_width = tf.to_float(initial_width) / ratio
      new_height = tf.to_float(initial_height) / ratio

      return tf.to_int32(new_width), tf.to_int32(new_height)

    def _useless(x, y):
      return x, y

    new_w, new_h = tf.cond(tf.logical_or(
      tf.greater(initial_width, tf.constant(self.image_size[1])),
      tf.greater(initial_height, tf.constant(self.image_size[0]))
    ),
      lambda: _resize(initial_width, initial_height),
      lambda: _useless(initial_width, initial_height))

    resized_image = tf.image.resize_images(image, [new_w, new_h])
    resized_image = tf.cast(resized_image, tf.uint8)
    return resized_image

  def distort_color(self, image, color_ordering=0, fast_mode=True, scope=None):
    with tf.name_scope(scope, 'distort_color', [image]):
      # if fast_mode:
      #   if color_ordering == 0:
      #     image = tf.image.random_brightness(image, max_delta=32. / 255.)
      #     image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      #   else:
      #     image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      #     image = tf.image.random_brightness(image, max_delta=32. / 255.)
      # else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        # image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      elif color_ordering == 1:
        # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        # image = tf.image.random_hue(image, max_delta=0.2)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        # image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      elif color_ordering == 3:
        # image = tf.image.random_hue(image, max_delta=0.2)
        # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

      # The random_* ops do not necessarily clamp.
      return tf.clip_by_value(image, 0.0, 1.0)

  def apply_with_random_selector(self, x, func, num_cases):
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    return control_flow_ops.merge([
      func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
      for case in range(num_cases)])[0]

  def input_parser_train(self, img_path, label):
    # convert the label to one-hot encoding
    # one_hot = tf.one_hot(label, len(self.selected_attrs))

    # read the img from file
    img = tf.map_fn(lambda x: tf.read_file(x), img_path)
    img_decoded = tf.map_fn(lambda x: tf.image.decode_jpeg(x, channels=3), img, dtype=tf.uint8)
    img_custom_resized = tf.map_fn(lambda x: self.custom_resize(x), img_decoded)
    img_resized = tf.map_fn(
      lambda x: tf.image.resize_image_with_crop_or_pad(x, self.image_size[0], self.image_size[1]),
      img_custom_resized)
    img_float = tf.map_fn(lambda x: tf.to_float(x), img_resized, dtype=tf.float32)
    img_normalized = tf.map_fn(lambda x: (1.0 / 255.0) * x, img_float)
    img_flipped = tf.map_fn(lambda x: tf.image.random_flip_left_right(x, SEED), img_normalized)
    img_final = tf.map_fn(lambda x: 2.0 * (x - 0.5), img_flipped)
    return img_final, label

  def input_parser_test(self, img_path, filename, label, fake_label):
    img = tf.map_fn(lambda x: tf.read_file(x), img_path)
    img_decoded = tf.map_fn(lambda x: tf.image.decode_jpeg(x, channels=3), img, dtype=tf.uint8)
    img_custom_resized = tf.map_fn(lambda x: self.custom_resize(x), img_decoded)
    img_resized = tf.map_fn(
      lambda x: tf.image.resize_image_with_crop_or_pad(x, self.image_size[0], self.image_size[1]),
      img_custom_resized)
    img_float = tf.map_fn(lambda x: tf.to_float(x), img_resized, dtype=tf.float32)
    img_normalized = tf.map_fn(lambda x: (2.0 / 255.0) * x - 1.0, img_float)
    return img_normalized, filename, label, fake_label

  def multi_label_list(self, label):
    y = [[1, 0, 0],  # black hair
         [0, 1, 0],  # blond hair
         [0, 0, 1]]  # brown hair
    label_list = []
    for i in range(4):
      fixed_label = label.copy()
      if i in [0, 1, 3]:  # Hair color to brown
        fixed_label[:3] = y[2]
      if i in [0, 2, 3]:  # Gender
        fixed_label[3] = 0 if fixed_label[3] == 1 else 1
      if i in [1, 2, 3]:  # Aged
        fixed_label[4] = 0 if fixed_label[4] == 1 else 1
      label_list.append(fixed_label)

    return label_list

  def single_label_list(self, label):
    # y = [[1, 0, 0],  # black hair
    #     [0, 1, 0],  # blond hair
    #     [0, 0, 1]]  # brown hair
    # choices = [c for c in range(len(self.selected_attrs))]# if c not in [8, 9, 11, 16]]
    # selected_choices = np.random.choice(choices, 10)
    label_list = []
    for i in range(len(self.selected_attrs)):
      fixed_label = label.copy()
      # iif i in [:
      #  fixed_label[:3] = y[i]
      # else:
      fixed_label[i] = 0 if fixed_label[i] == 1 else 1  # opposite value
      label_list.append(fixed_label)

    return label_list


def get_loader(dataset_path='../data/', image_size=(128, 128), batch_size=16, dataset='CelebA', mode='train',
               selected_attrs=('Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'), augmentation=False):
  if dataset == 'CelebA':
    metadata_path = os.path.join(dataset_path, 'list_attr_celebs.txt')
    image_path = os.path.join(dataset_path, 'celebA')
    output_path = os.path.join(dataset_path, 'dataset_path2')
    dataset = ImageDataset(image_path, metadata_path, mode, image_size, batch_size, output_path, selected_attrs, augmentation)
  elif dataset == "day_night":
    metadata_path = os.path.join(dataset_path, 'list_attrs.txt')
    image_path = os.path.join(dataset_path, 'images')
    output_path = os.path.join(dataset_path, 'dataset_path')
    dataset = ImageDataset(image_path, metadata_path, mode, image_size, batch_size, output_path, selected_attrs)
  elif dataset == 'RaFD':
    dataset = ImageFolder(image_path, transform)

  return dataset

def get_wild_images(image_path='../images/', image_size=(4032, 3024), batch_size=1,
               selected_attrs=('Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young')):
  output_path = os.path.join('../data/', 'dataset_path3')
  dataset = ImageFolder(image_path, image_size, batch_size, output_path, selected_attrs)

  return dataset

def get_all_images(dataset_path='../data/', image_size=(128, 128), batch_size=16, dataset='CelebA', mode='test',
               selected_attrs=('Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'), augmentation=False):
  metadata_path = os.path.join(dataset_path, 'list_attr_celebs.txt')
  image_path = os.path.join(dataset_path, 'celebA')
  output_path = os.path.join(dataset_path, 'dataset_path3')
  dataset = GenerationImageDataset(image_path, metadata_path, mode, image_size, batch_size, output_path, selected_attrs, augmentation)

  return dataset

def get_test_images(image_path='../images/', image_size=(128, 128), batch_size=1,
               selected_attrs=('Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young')):
  output_path = os.path.join('../data/', 'dataset_path2')
  dataset = TestImages(image_path, image_size, batch_size, output_path, selected_attrs)
  return dataset

if __name__ == '__main__':
  dataset = get_loader()

  # dataset = get_loader(FLAGS.dataset_path, config.image_size, config.batch_size, dataset='CelebA',
  #                      mode=('train' if FLAGS.train else 'test'))
  tensordict = {
    "init_ops": dataset.train_init,
    "input_pair": dataset.train_next
  }
  sess = tf.Session()

  with sess.as_default(), sess.graph.as_default():
    # sess.run(tf.global_variables_initializer())
    sess.run(dataset.train_init)

    i = 0
    while (1):
      try:
        nexty = sess.run(dataset.train_next)
        print(i)
        i += 1
      except tf.errors.OutOfRangeError:
        print("End of training dataset.")
        break


  print("end")

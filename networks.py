import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from PIL import Image, ImageDraw, ImageFont
from tools.tf_utils import gradient_summaries, lrelu, upsample, dense_upsample


class WNetwork():
  def __init__(self, config, input_pair, test_input_pair, is_training, global_step, nb_summaries_outputs):
    self.global_step = global_step
    self.learning_rate_global_step = tf.Variable(1, dtype=tf.int32, name='learning_rate_global_step', trainable=False)

    self.is_training = is_training
    self.nb_summaries_outputs = nb_summaries_outputs
    self.config = config
    self.selected_attrs = config.selected_attrs

    self.real_image, self.real_label = input_pair
    self.test_real_image, self.test_real_filename, self.test_real_label, self.test_fake_label_list = test_input_pair
    self.test_fake_label_list = [tf.squeeze(x, 1) for x in
                                 tf.split(self.test_fake_label_list, self.test_fake_label_list.shape[1].value, 1)]
    # rand_label_index = tf.random_uniform([tf.shape(self.real_label)[0]], 0, self.config.nd, dtype=tf.int32)
    # self.fake_label = tf.cast(tf.logical_xor(tf.cast(self.real_label, dtype=tf.bool),
    #                                          tf.cast(tf.one_hot(rand_label_index, self.config.nd), dtype=tf.bool)),
    #                           dtype=tf.int32)
    self.fake_label = tf.transpose(tf.random_shuffle(tf.transpose(self.real_label, [1, 0])), [1, 0])
    self.real_input = self.add_domain(self.real_image, self.fake_label)
    self.image_summaries = []
    self.summaries = []

    self.learning_rate = tf.train.polynomial_decay(self.config.lr, self.learning_rate_global_step,
                                                   self.config.num_iter_decay, 0.0, power=1)
    self.network_optimizer_d = config.network_optimizer(
      self.learning_rate, beta1=self.config.beta1, beta2=self.config.beta2, name='network_optimizer_d')
    self.network_optimizer_g = config.network_optimizer(
      self.learning_rate, beta1=self.config.beta1, beta2=self.config.beta2, name='network_optimizer_g')

    with tf.variable_scope('generator') as scope:
      self.fake_image = self.generator(self.real_input, self.real_image, self.real_label, self.fake_label, "generated")
      scope.reuse_variables()
      self.fake_input = self.add_domain(self.fake_image, self.real_label)
      self.reconstr_image = self.generator(self.fake_input, self.fake_image, self.fake_label, self.real_label,
                                           "reconstr")

      self.test_inputs = [self.add_domain(self.test_real_image, test_fake_label) for test_fake_label in
                          self.test_fake_label_list]

      # image_original_summary = tf.py_func(self.draw_label_on_image,
      #                                     [(image / 2.0 + 0.5) * 255, real_label, self.nb_summaries_outputs], tf.uint8)

      self.fixed_images = [self.generator(test_input, self.test_real_image, self.test_real_label,
                                          test_fake_label, "test") for test_input, test_fake_label in
                           zip(self.test_inputs, self.test_fake_label_list)]
      # images_generated_summary = tf.py_func(self.draw_images,
      #                                       [self.test_real_image, self.fixed_images, self.nb_summaries_outputs], tf.uint8)
      #
      self.image_summaries.append(
        tf.summary.image('test', tf.py_func(self.draw_label_on_image, [
          (tf.concat([self.test_real_image] + self.fixed_images, 2) / 2.0 + 0.5) * 255, self.test_real_label,
          self.test_fake_label_list], tf.uint8),
                         max_outputs=self.nb_summaries_outputs))

    with tf.variable_scope('discriminator') as scope:
      _, self.fake_src, self.fake_cls = self.discriminator(self.fake_image)
      scope.reuse_variables()
      self.second_to_last_src, self.real_src, self.real_cls = self.discriminator(self.real_image)
      self.second_to_last_src_, self.real_src_, self.real_cls_ = self.discriminator(self.real_image)

      epsilon = tf.random_uniform([], 0.0, 1.0)
      self.hat_image = epsilon * self.real_image + (1 - epsilon) * self.fake_image

      _, self.hat_src, self.hat_cls = self.discriminator(self.hat_image)

    self.build_losses()
    self.build_optim_ops()

  def add_domain(self, input, domain):
    new_domain = tf.expand_dims(tf.expand_dims(domain, 1), 1)
    new_domain = tf.tile(new_domain, [1, self.config.image_size[0], self.config.image_size[1], 1])
    output = tf.concat([input, tf.cast(new_domain, dtype=tf.float32)], 3)

    return output

  def generator(self, input, image, real_label, label, name):
    out = input

    with tf.variable_scope('conv'):
      for i, (kernel_size, stride, pad, nb_kernels) in enumerate(self.config.gen_conv_layers):
        out = layers.conv2d(out, num_outputs=nb_kernels, kernel_size=kernel_size,
                            stride=stride, activation_fn=None,
                            padding="VALID" if pad == 0 else "SAME",
                            biases_initializer=None,
                            variables_collections=tf.get_collection("generator"),
                            outputs_collections="activations", scope="conv_{}".format(i))

        out = tf.contrib.layers.instance_norm(out,
                                              center=True, scale=True,
                                              scope='in_{}'.format(i))
        out = tf.nn.relu(out)
        self.summaries.append(tf.contrib.layers.summarize_activation(out))

    with tf.variable_scope('residual'):
      for i, (kernel_size, stride, pad, nb_kernels) in enumerate(self.config.gen_res_layers):
        res = out
        res = layers.conv2d(res, num_outputs=nb_kernels, kernel_size=kernel_size,
                            stride=stride, activation_fn=None,
                            padding="VALID" if pad == 0 else "SAME",
                            biases_initializer=None,
                            variables_collections=tf.get_collection("generator"),
                            outputs_collections="activations", scope="conv1_{}".format(i))
        res = tf.contrib.layers.instance_norm(res,
                                              center=True, scale=True,
                                              scope='in1_{}'.format(i))
        res = tf.nn.relu(res)
        res = layers.conv2d(res, num_outputs=nb_kernels, kernel_size=kernel_size,
                            stride=stride, activation_fn=None,
                            padding="VALID" if pad == 0 else "SAME",
                            biases_initializer=None,
                            variables_collections=tf.get_collection("generator"),
                            outputs_collections="activations", scope="conv2_{}".format(i))
        res = tf.contrib.layers.instance_norm(res,
                                              center=True, scale=True,
                                              scope='in2_{}'.format(i))
        out = out + res
        self.summaries.append(tf.contrib.layers.summarize_activation(out))

    with tf.variable_scope("upsample"):
      for i, (kernel_size, stride, pad, nb_kernels) in enumerate(self.config.gen_upsample_layers):
        out = layers.conv2d_transpose(out, num_outputs=nb_kernels, kernel_size=kernel_size,
                                      stride=stride, activation_fn=None,
                                      biases_initializer=None,
                                      padding="VALID" if pad == 0 else "SAME",
                                      variables_collections=tf.get_collection("generator"),
                                      outputs_collections="activations", scope="upsample_{}".format(i))
        out = tf.contrib.layers.instance_norm(out,
                                              center=True, scale=True,
                                              scope='in_{}'.format(i))
        out = tf.nn.relu(out)
        self.summaries.append(tf.contrib.layers.summarize_activation(out))

    with tf.variable_scope("output"):
      (kernel_size, stride, pad, nb_kernels) = self.config.gen_output_layer
      out = layers.conv2d_transpose(out, num_outputs=nb_kernels, kernel_size=kernel_size,
                                    stride=stride, activation_fn=None,
                                    biases_initializer=None,
                                    padding="VALID" if pad == 0 else "SAME",
                                    variables_collections=tf.get_collection("generator"),
                                    outputs_collections="activations", scope="upsample_{}".format(i))
      out = tf.nn.tanh(out)
      self.summaries.append(tf.contrib.layers.summarize_activation(out))

      self.image_summaries.append(
        tf.summary.image('input_out', tf.concat([image, out], 2),
                         max_outputs=self.nb_summaries_outputs))
    return out

  def discriminator(self, input):
    out = input
    with tf.variable_scope('conv'):
      for i, (kernel_size, stride, pad, nb_kernels) in enumerate(self.config.disc_conv_layers):
        out = layers.conv2d(out, num_outputs=nb_kernels, kernel_size=kernel_size,
                            stride=stride, activation_fn=None,
                            padding="VALID" if pad == 0 else "SAME",
                            variables_collections=tf.get_collection("discriminator"),
                            outputs_collections="activations", scope="conv_{}".format(i))
        out = tf.nn.leaky_relu(out, alpha=0.01)
        if self.config.WGAN_CT and i >= 3:
          out = tf.nn.dropout(out, keep_prob=0.50)
        self.summaries.append(tf.contrib.layers.summarize_activation(out))

      with tf.variable_scope('src'):
        (kernel_size, stride, pad, nb_kernels) = self.config.disc_output_layer_src
        out_src = layers.conv2d(out, num_outputs=nb_kernels, kernel_size=kernel_size,
                                stride=stride, activation_fn=None,
                                biases_initializer=None,
                                padding="VALID" if pad == 0 else "SAME",
                                variables_collections=tf.get_collection("discriminator"),
                                outputs_collections="activations", scope="conv_{}".format(i))

      with tf.variable_scope('cls'):
        (kernel_size, stride, pad, nb_kernels) = self.config.disc_ouput_layer_cls
        if self.config.fc_cls:
            out = layers.flatten(out)
            out_cls = layers.fully_connected(out, num_outputs=nb_kernels,
                                   activation_fn=None,
                                   variables_collections=tf.get_collection("discriminator"),
                                   outputs_collections="activations", scope="fc_{}".format(i))
        else:
            out_cls = layers.conv2d(out, num_outputs=nb_kernels, kernel_size=kernel_size,
                                    stride=stride, activation_fn=None,
                                    biases_initializer=None,
                                    padding="VALID" if pad == 0 else "SAME",
                                    variables_collections=tf.get_collection("discriminator"),
                                    outputs_collections="activations", scope="conv_{}".format(i))
            out_cls = tf.squeeze(tf.squeeze(out_cls, 1), 1)

      return layers.flatten(out), layers.flatten(out_src), out_cls

  def build_losses(self):
    # Gradient penalty
    ddx = tf.gradients(self.hat_src, self.hat_image)[0]
    ddx_shape = ddx.get_shape()
    ddx = tf.reshape(ddx, [-1, np.prod([x.value for x in ddx_shape][1:])])
    ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
    self.gp_loss = tf.reduce_mean(tf.square(ddx - 1.0))
    self.d_loss = 0
    # Consistancy term
    if self.config.WGAN_CT:
      CT = self.config.lambda_ct * tf.reduce_mean(tf.square(self.real_src - self.real_src_), axis=1)
      CT += self.config.lambda_ct * 0.1 * tf.reduce_mean(tf.square(self.second_to_last_src - self.second_to_last_src_), axis=1)
      CT_ = tf.maximum(CT - self.config.factor_m, 0.0 * (CT - self.config.factor_m))
      self.d_ct_loss = tf.reduce_mean(CT_)
      self.d_loss += self.d_ct_loss

    self.d_loss_real = -tf.reduce_mean(self.real_src)
    self.d_loss_fake = tf.reduce_mean(self.fake_src)
    self.d_loss_gp = self.config.lambda_gp * self.gp_loss
    self.real_loss_cls = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
      logits=self.real_cls,
      labels=tf.cast(self.real_label, dtype=tf.float32)))
    self.d_loss_cls = self.config.lambda_cls * self.real_loss_cls

    # ------------------------------------ D ---------------------------------------
    self.d_vars = tf.get_collection("variables", "discriminator")
    self.d_loss += self.d_loss_real + self.d_loss_fake + self.d_loss_gp + self.d_loss_cls

    # ------------------------------------ G ---------------------------------------
    self.fake_loss_cls = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
      logits=self.fake_cls,
      labels=tf.cast(self.fake_label, dtype=tf.float32)))

    self.reconstr_loss = tf.reduce_mean(tf.abs(self.real_image - self.reconstr_image))
    self.g_loss_fake = -tf.reduce_mean(self.fake_src)
    self.g_loss_cls = self.config.lambda_cls * self.fake_loss_cls
    self.g_loss_rec = self.config.lambda_rec * self.reconstr_loss
    self.g_vars = tf.get_collection("variables", "generator")
    self.g_loss = self.g_loss_fake + self.g_loss_cls + self.g_loss_rec

  def build_optim_ops(self):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      d_gradients = self.network_optimizer_d.compute_gradients(self.d_loss, var_list=self.d_vars)
      d_clipped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) if grad is not None else (grad, var) for
                             grad, var in d_gradients]
      self.d_train = self.network_optimizer_d.apply_gradients(d_clipped_gradients)

      g_gradients = self.network_optimizer_g.compute_gradients(self.g_loss, var_list=self.g_vars)
      g_clipped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) if grad is not None else (grad, var) for grad, var
                             in g_gradients]
      self.g_train = self.network_optimizer_g.apply_gradients(g_clipped_gradients)

    self.d_loss_summaries = [tf.summary.scalar('d_loss', self.d_loss),
                             tf.summary.scalar('d_loss_gp', self.d_loss_gp),
                             tf.summary.scalar('d_loss_real', self.d_loss_real),
                             tf.summary.scalar('d_loss_fake', self.d_loss_fake),
                             tf.summary.scalar('g_loss_fake', self.g_loss_fake),
                             tf.summary.scalar('d_loss_cls', self.d_loss_cls),
                             tf.summary.scalar("d_lr", self.learning_rate)]
    self.g_loss_summaries = [tf.summary.scalar('g_loss', self.g_loss),
                             tf.summary.scalar('g_loss_fake', self.g_loss_fake),
                             tf.summary.scalar('g_loss_cls', self.g_loss_cls),
                             tf.summary.scalar('g_loss_rec', self.g_loss_rec),
                             tf.summary.scalar("lr", self.learning_rate)]

    self.g_merged_summary = tf.summary.merge(self.summaries + self.g_loss_summaries +
                                             [gradient_summaries(g_clipped_gradients)])
    self.d_merged_summary = tf.summary.merge(self.summaries + self.d_loss_summaries +
                                             [gradient_summaries(d_clipped_gradients)])

    self.test_summaries = tf.summary.merge(self.image_summaries)

  def draw_label_on_image(self, imgs, real_label, fake_labels):
    n, h, w, c = imgs.shape
    outputs = np.zeros((n, h, w, c), dtype=np.uint8)
    fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 12)
    outputs_imgs = []
    labels = [real_label] + list(fake_labels)
    for i in range(n):
      outputs[i] = (imgs[i])[:, :, :].astype(np.uint8)
      outputs_img = Image.fromarray(outputs[i], mode='RGB')
      size = outputs_img.size
      txt = Image.new('RGB', (size[0], size[1] * 5), (0, 0, 0))
      dr = ImageDraw.Draw(txt)
      img_labels = [label[i] for label in labels]
      # print(img_labels)
      j = 0
      for test_label in img_labels:
        for ind, label in enumerate(test_label):
          if label == 1:
            dr.text((j, 12 * ind), self.selected_attrs[ind], font=fnt, fill=(255, 255, 255))
          else:
            dr.text((j, 12 * ind), self.selected_attrs[ind], font=fnt, fill=(255, 0, 0))
        j += 128
      outputs_imgs.append(np.concatenate((txt, outputs_img), 0))

    return np.stack(outputs_imgs, axis=0)

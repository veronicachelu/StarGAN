import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import dataset
from PIL import Image
from tools.tf_utils import lrelu, define_saver
import os
import time
import copy
from tools.timer import Timer
from PIL import Image, ImageDraw, ImageFont
FLAGS = tf.app.flags.FLAGS

class GAN:
  def __init__(self, config, global_step, tensor_dict, is_training, nb_summaries_outputs):
    self.is_training = is_training
    self.config = config
    self.global_step = global_step
    self.init_ops = tensor_dict["init_ops"]
    self.input_pair = tensor_dict["input_pair"]
    self.test_input_pair = tensor_dict["test_input_pair"]
    self.optimizer = config.network_optimizer

    if self.is_training:
      self.model_path = os.path.join(config.logdir, "models")
      self.summary_path = os.path.join(config.logdir, "summaries")
      tf.gfile.MakeDirs(self.model_path)
      tf.gfile.MakeDirs(self.summary_path)

      self.increment_global_step = self.global_step.assign_add(1)

      self.summary_writer = tf.summary.FileWriter(self.summary_path)
      self.network = config.network(config, self.input_pair, self.test_input_pair, self.is_training, self.global_step, nb_summaries_outputs)
      self.increment_learning_rate_global_step = self.network.learning_rate_global_step.assign_add(1)
      self.saver = self.loader = define_saver(exclude=(r'.*_temporary/.*',))
    else:
      self.summary_path = os.path.join(config.logdir, "results")
      tf.gfile.MakeDirs(self.summary_path)

      self.network = config.network(config, self.input_pair, self.test_input_pair, self.is_training, self.global_step,
                                    nb_summaries_outputs)

      self.loader = define_saver(exclude=(r'.*_temporary/.*',))

  def train(self, sess):
    # _t = {'Discr_all': Timer(), "Discr_one": Timer(), 'Gen_one': Timer(), 'Summaries': Timer()}
    with sess.as_default(), sess.graph.as_default():
      # sess.run(tf.global_variables_initializer())
      sess.run(self.init_ops)
      tf.logging.info("Training...")
      step = sess.run(self.global_step)
      # start_time = time.time()
      while step <= self.config.max_iters:
        # tf.logging.info(step)
        # _t['Discr_all'].tic()
        _ = sess.run([self.network.d_train])

        # for _ in range(0, self.config.d_iters):
        #   # _t['Discr_one'].tic()
        #   _ = sess.run([self.network.d_train])
        #   # _t['Discr_one'].toc()
        # # _t['Discr_all'].toc()
        #
        # # _t['Gen_one'].tic()
        if step % self.config.d_train_repeat == 0:
          _ = sess.run([self.network.g_train])
        # _t['Gen_one'].toc()

        if step % self.config.summary_every == 0:
          # _t['Summaries'].tic()
          test_summaries, d_summaries, g_summaries = sess.run([self.network.test_summaries, self.network.d_merged_summary, self.network.g_merged_summary])
          # _t['Summaries'].toc()
          self.summary_writer.add_summary(test_summaries, step)
          self.summary_writer.add_summary(d_summaries, step)
          self.summary_writer.add_summary(g_summaries, step)
          # tf.logging.info('Discr_all time is %f' % _t['Discr_all'].average_time)
          # tf.logging.info('Discr_one time is %f' % _t['Discr_one'].average_time)
          # tf.logging.info('Gen_one time is %f' % _t['Gen_one'].average_time)
          # tf.logging.info('Summaries time is %f' % _t['Summaries'].average_time)
        if step % self.config.checkpoint_every == 0:
          self.saver.save(sess, self.model_path + '/model-' + str(step) + '.cptk',
                     global_step=self.global_step)
          tf.logging.info("Saved Model at {}".format(self.model_path + '/model-' + str(step) + '.cptk'))

          # self.get_time_info(start_time, step)

        sess.run(self.increment_global_step)
        step += 1

        if step > self.config.num_iter_decay:
          tf.logging.info("decayed")
          sess.run(self.increment_learning_rate_global_step)

  def generate(self, sess, nb_summaries_outputs):
    with sess.as_default(), sess.graph.as_default():
      sess.run(self.init_ops)
      tf.logging.info("Generating results...")
      count = 0
      tf_images = (tf.concat([self.network.test_real_image] + self.network.fixed_images, 2) / 2.0 + 0.5) * 255
      while True:
        try:
          images = sess.run(tf_images)
          for image in images:
            Image.fromarray(image.astype(np.uint8)).save(os.path.join(self.summary_path, "image_{}.png".format(count)))
            count += 1
        except tf.errors.OutOfRangeError:
          tf.logging.info("End of training dataset.")
          break

  def test_inference_time(self, sess):
    _t = {'inference': Timer()}
    with sess.as_default(), sess.graph.as_default():
      sess.run(self.init_ops)
      for i in range(1000):
        _t['inference'].tic()
        sess.run(self.network.fixed_images[0])
        _t['inference'].toc()
        tf.logging.info('inference time is %f' % _t['inference'].average_time)
    # tf.logging.info('inference time is %f' % _t['inference'].average_time)

  def run_wild(self, sess):
    _t = {'inference': Timer()}
    self.wild_results = os.path.join(self.config.logdir, "wild_results")
    tf.gfile.MakeDirs(self.wild_results)
    fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 12)
    with sess.as_default(), sess.graph.as_default():
      sess.run(self.init_ops)
      tf.logging.info("Generating results...")
      count = 0
      tf_images = (tf.concat([self.network.test_real_image] + self.network.fixed_images, 2) / 2.0 + 0.5) * 255
      while True:
        try:
          images = sess.run(tf_images)

          for image in images:
            outputs_img = Image.fromarray(image.astype(np.uint8), mode='RGB')
            size = outputs_img.size
            txt = Image.new('RGB', (size[0], size[1]), (0, 0, 0))
            dr = ImageDraw.Draw(txt)

            dr.text((0, 60), "original", font=fnt, fill=(255, 255, 255))
            j = 128
            for ind, label in enumerate(self.config.selected_attrs):
              dr.text((j, 60), label, font=fnt, fill=(255, 255, 255))
              j += 128
            rez = np.concatenate((txt, outputs_img), 0)
            rez = Image.fromarray(rez.astype(np.uint8), mode='RGB')
            outputs_img.save(os.path.join(self.wild_results, "image_{}.png".format(count)))
            count += 1
        except tf.errors.OutOfRangeError:
          tf.logging.info("End of training dataset.")
          break

  def generate_all(self, sess):
    self.generated_results = os.path.join(self.config.logdir, "generated_results")
    tf.gfile.MakeDirs(self.generated_results)
    with sess.as_default(), sess.graph.as_default():
      sess.run(self.init_ops)
      tf.logging.info("Generating results...")
      while True:
        try:
          image_filenames_eval, generated_images_eval = sess.run([self.network.test_real_filename, self.network.fixed_images])
          generated_images_eval = np.swapaxes([np.split(i, self.config.batch_size, axis=0) for i in generated_images_eval], 0, 1)
          generated_images_eval = [[np.squeeze(i, 0) for i in f] for f in generated_images_eval]

          for f, i in zip(image_filenames_eval, generated_images_eval):
            for counter, j in enumerate(self.config.selected_attrs):
              rez = Image.fromarray(((i[counter] / 2.0 + 0.5) * 255).astype(np.uint8))
              root, ext = os.path.splitext(f)
              rez.save(os.path.join(self.generated_results, "{}_{}{}".format(root.decode("utf-8"), j, ".jpeg")))

        except tf.errors.OutOfRangeError:
          tf.logging.info("End of training dataset.")
          break

  def generate_original(self, sess):
    self.generated_results = os.path.join(self.config.logdir, "generated_results_originals")
    tf.gfile.MakeDirs(self.generated_results)
    with sess.as_default(), sess.graph.as_default():
      sess.run(self.init_ops)
      tf.logging.info("Generating results...")
      while True:
        try:
          original_image_filenames, original_images = sess.run([self.network.test_real_filename, self.network.test_real_image])
          # generated_images_eval = np.swapaxes([np.split(i, self.config.batch_size, axis=0) for i in generated_images_eval], 0, 1)
          # generated_images_eval = [[np.squeeze(i, 0) for i in f] for f in generated_images_eval]

          for f, i in zip(original_image_filenames, original_images):
            rez = Image.fromarray(((i / 2.0 + 0.5) * 255).astype(np.uint8))
            root, ext = os.path.splitext(f)
            rez.save(os.path.join(self.generated_results, "{}{}".format(root.decode("utf-8"), ".jpeg")))

        except tf.errors.OutOfRangeError:
          tf.logging.info("End of training dataset.")
          break

  def get_time_info(self, start_time, current_it):
    elapsed_time = time.time() - start_time
    em, es = divmod(elapsed_time, 60)
    eh, em = divmod(em, 60)
    remaining_time = int((self.config.max_iters - current_it) / current_it * elapsed_time)
    rm, rs = divmod(remaining_time, 60)
    rh, rm = divmod(rm, 60)

    tf.logging.info("Elapsed time (h:m:s): {}. Remaining time (h:m:s): {}".format(
      str(int(eh)) + ':' + str(int(em)) + ':' + str(int(es)), str(int(rh)) + ':' + str(int(rm)) + ':' + str(int(rs))))

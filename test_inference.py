import os
import tensorflow as tf
import tools
from tools import tf_utils
import configs
from dataset import get_loader, get_wild_images, get_test_images
FLAGS = tf.app.flags.FLAGS

def test_inference(config, logdir):
  tf.reset_default_graph()
  sess = tf.Session()
  with sess:
    # with tf.device("/gpu:0"):
    with config.unlocked:
      config.logdir = logdir
      config.network_optimizer = getattr(tf.train, config.network_optimizer)
      global_step = tf.Variable(1, dtype=tf.int32, name='global_step', trainable=False)
      if FLAGS.dataset_type == "test":
        dataset = get_test_images(FLAGS.images_path, config.image_size, 1, selected_attrs=config.selected_attrs)
      else:
        dataset = get_wild_images(FLAGS.images_path, config.image_size, 1, selected_attrs=config.selected_attrs)
      tensordict = {
        "init_ops": [dataset.train_init, dataset.test_init],
        "input_pair": dataset.train_next,
        "test_input_pair": dataset.test_next,
      }
      model = config.model(config, global_step, tensordict, False, 500)

      sess.run(tf.global_variables_initializer())
      ckpt = tf.train.get_checkpoint_state(os.path.join(FLAGS.load_from, "models"))
      print("Loading Model from {}".format(ckpt.model_checkpoint_path))
      model.loader.restore(sess, ckpt.model_checkpoint_path)
      sess.run(tf.local_variables_initializer())

      model.test_inference_time(sess)

def main(_):
  tf_utils.set_up_logging()
  os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU
  if not FLAGS.config:
    raise KeyError('You must specify a configuration.')
  if FLAGS.load_from:
    logdir = FLAGS.logdir = FLAGS.load_from
  else:
    if FLAGS.logdir and os.path.exists(FLAGS.logdir):
      run_number = [int(f.split("-")[0]) for f in os.listdir(FLAGS.logdir) if
                    os.path.isdir(os.path.join(FLAGS.logdir, f)) and FLAGS.config in f]
      run_number = max(run_number) + 1 if len(run_number) > 0 else 0
    else:
      run_number = 0
    logdir = FLAGS.logdir and os.path.expanduser(os.path.join(
      FLAGS.logdir, '{}-{}'.format(run_number, FLAGS.config)))
  try:
    config = tf_utils.load_config(logdir)
  except IOError:
    config = tools.AttrDict(getattr(configs, FLAGS.config)())
    config = tf_utils.save_config(config, logdir)
  test_inference(config, logdir)

if __name__ == '__main__':
  tf.app.flags.DEFINE_string(
    'logdir', "./logdir",
    'Base directory to store logs.')
  tf.app.flags.DEFINE_string(
    'config', "simple",
    'Configuration to execute.')
  tf.app.flags.DEFINE_string(
    # 'load_from', None,
    'load_from', "./logdir/5-simple",
    'Load directory to load models from.')
  tf.app.flags.DEFINE_string(
    'GPU', "0",
    """The GPU device to run on""")
  tf.app.flags.DEFINE_string(
    'dataset_type', "test",
    """What images to run on""")
  tf.app.flags.DEFINE_string(
    'images_path', "../images/",
    """The GPU device to run on""")
  tf.app.run()

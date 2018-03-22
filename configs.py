import functools
import models
import networks


def default():
  model = models.GAN
  network = networks.WNetwork
  nd = 5
  image_size = [128, 128]
  gen_conv_layers = (7, 1, 3, 64), (4, 2, 1, 128), (4, 2, 1, 256)  # , (4, 2, 0, 100)
  gen_res_layers = (3, 1, 1, 256), (3, 1, 1, 256), (3, 1, 1, 256), (3, 1, 1, 256), (3, 1, 1, 256), (3, 1, 1, 256)
  gen_upsample_layers = (4, 2, 1, 128), (4, 2, 1, 64),
  gen_output_layer = (7, 1, 3, 3)

  disc_conv_layers = (4, 2, 1, 64), (4, 2, 1, 128), (4, 2, 1, 256), (4, 2, 1, 512), (4, 2, 1, 1024), (4, 2, 1, 2048)
  disc_output_layer_src = (3, 1, 1, 1)
  disc_ouput_layer_cls = (2, 1, 0, nd)

  lr = 0.0001
  lambda_cls = 1
  lambda_rec = 10
  lambda_gp = 10
  # std = 0.02
  max_iters = 12538 * 20
  max_epochs = 20
  num_iter_decay = 12538 * 10
  batch_size = 16
  summary_every = 500
  checkpoint_every = 500
  network_optimizer = 'AdamOptimizer'
  beta1 = 0.5
  beta2 = 0.999
  d_iters = 5
  selected_attrs = ('Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young')

  return locals()


def all_attrs():
  locals().update(default())
  nd = 40
  image_size = [128, 128]
  gen_conv_layers = (7, 1, 3, 64), (4, 2, 1, 128), (4, 2, 1, 256)  # , (4, 2, 0, 100)
  gen_res_layers = (3, 1, 1, 256), (3, 1, 1, 256), (3, 1, 1, 256), (3, 1, 1, 256), (3, 1, 1, 256), (3, 1, 1, 256)
  gen_upsample_layers = (4, 2, 1, 128), (4, 2, 1, 64),
  gen_output_layer = (7, 1, 3, 3)

  disc_conv_layers = (4, 2, 1, 64), (4, 2, 1, 128), (4, 2, 1, 256), (4, 2, 1, 512), (4, 2, 1, 1024), (4, 2, 1, 2048)
  disc_output_layer_src = (3, 1, 1, 1)
  disc_ouput_layer_cls = (2, 1, 0, nd)
  max_iters = 202599 * 20
  num_iter_decay = 202599 * 10
  selected_attrs = ('5_o_Clock_Shadow', 'Arched_Eyebrows',
                    'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
                    'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee',
                    'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache',
                    'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
                    'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
                    'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young')
  return locals()

def relevant():
  locals().update(default())
  nd = 10
  image_size = [128, 128]
  gen_conv_layers = (7, 1, 3, 64), (4, 2, 1, 128), (4, 2, 1, 256)  # , (4, 2, 0, 100)
  gen_res_layers = (3, 1, 1, 256), (3, 1, 1, 256), (3, 1, 1, 256), (3, 1, 1, 256), (3, 1, 1, 256), (3, 1, 1, 256)
  gen_upsample_layers = (4, 2, 1, 128), (4, 2, 1, 64),
  gen_output_layer = (7, 1, 3, 3)

  disc_conv_layers = (4, 2, 1, 64), (4, 2, 1, 128), (4, 2, 1, 256), (4, 2, 1, 512), (4, 2, 1, 1024), (4, 2, 1, 2048)
  disc_output_layer_src = (3, 1, 1, 1)
  disc_ouput_layer_cls = (2, 1, 0, nd)
  max_iters = 202599 * 20
  num_iter_decay = 202599 * 10
  d_train_repeat = 5

  selected_attrs = ('Bangs', 'Eyeglasses', 'Goatee', 'Heavy_Makeup', 'Mouth_Slightly_Open', 'Mustache',
   'Narrow_Eyes', 'No_Beard', 'Wearing_Hat', 'Wearing_Lipstick')
  summary_every = 10000
  checkpoint_every = 10000
  WGAN_CT = False
  lambda_ct = 2.0
  factor_m = 0.0
  augmentation = True

  return locals()

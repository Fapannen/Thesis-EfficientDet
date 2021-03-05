import efficientdet_keras as effdet
import hparams_config
import train
import utils
import tensorflow as tf
import dataloader
import numpy as np
from PIL import Image

class CustomEfficientDetModel(effdet.EfficientDetNet):
  """EfficientDet full keras model with pre and post processing."""

  def _preprocessing(self, raw_images, image_size, mode=None):
    """Preprocess images before feeding to the network."""
    if not mode:
      return raw_images, None

    image_size = utils.parse_image_size(image_size)
    if mode != 'infer':
      # We only support inference for now.
      raise ValueError('preprocessing must be infer or empty')

    def map_fn(image):
      input_processor = dataloader.DetectionInputProcessor(
          image, image_size)
      input_processor.normalize_image()
      input_processor.set_scale_factors_to_output_size()
      image = input_processor.resize_and_crop_image()
      image_scale = input_processor.image_scale_to_original
      return image, image_scale

    """
    if raw_images.shape.as_list()[0]:  # fixed batch size.
      batch_size = raw_images.shape.as_list()[0]
      outputs = [map_fn(raw_images[i]) for i in range(batch_size)]
      return [tf.stack(y) for y in zip(*outputs)]
	"""
    # otherwise treat it as dynamic batch size.
    return tf.vectorized_map(map_fn, raw_images)

  def _postprocess(self, cls_outputs, box_outputs, scales, mode=None):
    """Postprocess class and box predictions."""
    if not mode:
      return cls_outputs, box_outputs

    if mode == 'global':
      return postprocess.postprocess_global(self.config.as_dict(), cls_outputs,
                                            box_outputs, scales)
    if mode == 'per_class':
      return postprocess.postprocess_per_class(self.config.as_dict(),
                                               cls_outputs, box_outputs, scales)
    raise ValueError('Unsupported postprocess mode {}'.format(mode))

  def call(self, inputs, training=False, pre_mode='infer', post_mode='global'):
    """Call this model.
    Args:
      inputs: a tensor with common shape [batch, height, width, channels].
      training: If true, it is training mode. Otherwise, eval mode.
      pre_mode: preprocessing mode, must be {None, 'infer'}.
      post_mode: postprrocessing mode, must be {None, 'global', 'per_class'}.
    Returns:
      the output tensor list.
    """
    config = self.config

    # preprocess.
    inputs, scales = self._preprocessing(inputs, config.image_size, pre_mode)
    # network.
    outputs = super().call(inputs, training)
    return outputs

"""
    if 'object_detection' in config.heads and post_mode:
      # postprocess for detection
      det_outputs = self._postprocess(outputs[0], outputs[1], scales, post_mode)
      outputs = det_outputs + outputs[2:]
"""

"""
imgs = [np.array(Image.open('people.jpg'))]
# Create model config.
config = hparams_config.get_efficientdet_config('efficientdet-d0')
config.is_training_bn = False
config.image_size = '1920x1280'
config.nms_configs.score_thresh = 0.4
config.nms_configs.max_output_size = 100

# Use 'mixed_float16' if running on GPUs.
policy = tf.keras.mixed_precision.Policy('float32')
tf.keras.mixed_precision.set_global_policy(policy)

# Create and run the model.
model = CustomEfficientDetModel(config=config)
model.build((None, None, None, 3))
model.load_weights(tf.train.latest_checkpoint("../../../thesis/d0/efficientdet-d0"))
model.summary()
"""

config = hparams_config.get_efficientdet_config('efficientdet-d0')
config.image_size = '1920x1080'
config.image_size = utils.parse_image_size(config.image_size)

config.steps_per_execution = 5717
config.batch_size = 64
config.num_epochs = 50
config.steps_per_epoch = 5717

img = np.array(Image.open('people.jpg'))

model = CustomEfficientDetModel(config=config)
model = train.setup_model(model, config)


import os

import ast
import tensorflow.compat.v1 as tf
import tf_slim as slim

from magenta.models.arbitrary_image_stylization import arbitrary_image_stylization_build_model as build_model
from magenta.models.image_stylization import image_utils

from art_style_transfer import imagenette_data

DEFAULT_CONTENT_WEIGHTS = "{'vgg_16/conv3': 1.0}"
DEFAULT_STYLE_WEIGHTS = ('{"vgg_16/conv1": 0.5e-3, "vgg_16/conv2": 0.5e-3,'
                         ' "vgg_16/conv3": 0.5e-3, "vgg_16/conv4": 0.5e-3}')

flags = tf.app.flags
flags.DEFINE_float('clip_gradient_norm', 0, 'Clip gradients to this norm')
flags.DEFINE_float('learning_rate', 1e-5, 'Learning rate')
flags.DEFINE_float('total_variation_weight', 1e4, 'Total variation weight')
flags.DEFINE_string('content_weights', DEFAULT_CONTENT_WEIGHTS,
                                        'Content weights')
flags.DEFINE_string('style_weights', DEFAULT_STYLE_WEIGHTS, 'Style weights')
flags.DEFINE_integer('batch_size', 8, 'Batch size.')
flags.DEFINE_integer('image_size', 256, 'Image size.')
flags.DEFINE_boolean('random_style_image_size', True,
                                         'Wheather to augment style images or not.')
flags.DEFINE_boolean(
        'augment_style_images', True,
        'Wheather to resize the style images to a random size or not.')
flags.DEFINE_boolean('center_crop', False,
                                         'Wheather to center crop the style images.')
flags.DEFINE_integer('ps_tasks', 0,
                                         'Number of parameter servers. If 0, parameters '
                                         'are handled locally by the worker.')
flags.DEFINE_integer('save_summaries_secs', 15,
                                         'Frequency at which summaries are saved, in seconds.')
flags.DEFINE_integer('save_interval_secs', 15,
                                         'Frequency at which the model is saved, in seconds.')
flags.DEFINE_integer('task', 0, 'Task ID. Used when training with multiple '
                                         'workers to identify each worker.')
flags.DEFINE_integer('train_steps', 8000000, 'Number of training steps.')
flags.DEFINE_string('master', '', 'BNS name of the TensorFlow master to use.')
flags.DEFINE_string('style_dataset_file', None, 'Style dataset file.')
flags.DEFINE_string('train_dir', None,
                                        'Directory for checkpoints and summaries.')
flags.DEFINE_string('inception_v3_checkpoint', None,
                                        'Path to the pre-trained inception_v3 checkpoint.')

FLAGS = flags.FLAGS

FLAGS.batch_size=2
# FLAGS.imagenet_data_dir="/home/neolaw_gmail_com/tensorflow_datasets/imagenette/full-size-v2/1.0.0/"
FLAGS.imagenet_data_dir="/mnt/disks/gpu-one-101/imagenette2/train/"
FLAGS.vgg_checkpoint="/home/neolaw_gmail_com/projects/art-hackathon-21/data/arbitrary_style_transfer/model.ckpt" 
FLAGS.inception_v3_checkpoint="not-using" 
FLAGS.style_dataset_file="/mnt/disks/gpu-one-101/d2d_cobwebbed-.tfrecord" 
FLAGS.train_dir="/home/neolaw_gmail_com/projects/art-hackathon-21/logdir/traindir/" 
FLAGS.content_weights="{'vgg_16/conv3':2.0}"
FLAGS.random_style_image_size=False 
FLAGS.augment_style_images=False 
FLAGS.center_crop=True 
FLAGS.logtostderr=True

def mock_imagenet_inputs(batch_size, image_size, num_readers=1,
                                        num_preprocess_threads=4):
    """Loads a batch of imagenet inputs.
    Used as a replacement for inception.image_processing.inputs in
    tensorflow/models in order to get around the use of hard-coded flags in the
    image_processing module.
    Args:
        batch_size: int, batch size.
        image_size: int. The images will be resized bilinearly to shape
                [image_size, image_size].
        num_readers: int, number of preprocessing threads per tower.    Must be a
                multiple of 4.
        num_preprocess_threads: int, number of parallel readers.
    Returns:
        4-D tensor of images of shape [batch_size, image_size, image_size, 3], with
        values in [0, 1].
    Raises:
        IOError: If ImageNet data files cannot be found.
        ValueError: If `num_preprocess_threads is not a multiple of 4 or
                `num_readers` is less than 1.
    """
    imagenet = imagenette_data.ImagenetData('train')

    with tf.name_scope('batch_processing'):
        data_files = imagenet.data_files()
        if data_files is None:
            raise IOError('No ImageNet data files found')

        # Create filename_queue.
        filename_queue = tf.train.string_input_producer(data_files,
                                                                                                        shuffle=True,
                                                                                                        capacity=16)

        if num_preprocess_threads % 4:
            raise ValueError('Please make num_preprocess_threads a multiple '
                                             'of 4 (%d %% 4 != 0).' % num_preprocess_threads)

        if num_readers < 1:
            raise ValueError('Please make num_readers at least 1')

        # Approximate number of examples per shard.
        examples_per_shard = 1024
        # Size the random shuffle queue to balance between good global
        # mixing (more examples) and memory use (fewer examples).
        # 1 image uses 299*299*3*4 bytes = 1MB
        # The default input_queue_memory_factor is 16 implying a shuffling queue
        # size: examples_per_shard * 16 * 1MB = 17.6GB
        input_queue_memory_factor = 16
        min_queue_examples = examples_per_shard * input_queue_memory_factor
        examples_queue = tf.RandomShuffleQueue(
                capacity=min_queue_examples + 3 * batch_size,
                min_after_dequeue=min_queue_examples,
                dtypes=[tf.string])

        # Create multiple readers to populate the queue of examples.
        enqueue_ops = []
        for _ in range(num_readers):
            reader = imagenet.reader()
            _, value = reader.read(filename_queue)
            enqueue_ops.append(examples_queue.enqueue([value]))

        tf.train.queue_runner.add_queue_runner(
                tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
        example_serialized = examples_queue.dequeue()

        images_and_labels = []
        for _ in range(num_preprocess_threads):
            # Parse a serialized Example proto to extract the image and metadata.
            image_buffer, label_index, _, _ = image_utils._parse_example_proto(
                    example_serialized)
            image = tf.image.decode_jpeg(image_buffer, channels=3)

            # pylint: disable=protected-access
            image = image_utils._aspect_preserving_resize(image, image_size + 2)
            image = image_utils._central_crop([image], image_size, image_size)[0]
            # pylint: enable=protected-access
            image.set_shape([image_size, image_size, 3])
            image = tf.to_float(image) / 255.0

            images_and_labels.append([image, label_index])

        images, label_index_batch = tf.train.batch_join(
                images_and_labels,
                batch_size=batch_size,
                capacity=2 * num_preprocess_threads * batch_size)

        images = tf.reshape(images, shape=[batch_size, image_size, image_size, 3])

        # Display the training images in the visualizer.
        tf.summary.image('images', images)

        return images, tf.reshape(label_index_batch, [batch_size])


def new_main(unused_argv=None):
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        # Forces all input processing onto CPU in order to reserve the GPU for the
        # forward inference and back-propagation.
        device = '/cpu:0' if not FLAGS.ps_tasks else '/job:worker/cpu:0'
        with tf.device(
                tf.train.replica_device_setter(FLAGS.ps_tasks, worker_device=device)):
            # Loads content images.
            content_inputs_, _ = mock_imagenet_inputs(FLAGS.batch_size, FLAGS.image_size)

            # Loads style images.
            [style_inputs_, _,
             style_inputs_orig_] = image_utils.arbitrary_style_image_inputs(
                     FLAGS.style_dataset_file,
                     batch_size=FLAGS.batch_size,
                     image_size=FLAGS.image_size,
                     shuffle=True,
                     center_crop=FLAGS.center_crop,
                     augment_style_images=FLAGS.augment_style_images,
                     random_style_image_size=FLAGS.random_style_image_size)

        with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
            # Process style and content weight flags.
            content_weights = ast.literal_eval(FLAGS.content_weights)
            style_weights = ast.literal_eval(FLAGS.style_weights)

            # Define the model
            stylized_images, total_loss, loss_dict, _ = build_model.build_model(
                    content_inputs_,
                    style_inputs_,
                    trainable=True,
                    is_training=True,
                    inception_end_point='Mixed_6e',
                    style_prediction_bottleneck=100,
                    adds_losses=True,
                    content_weights=content_weights,
                    style_weights=style_weights,
                    total_variation_weight=FLAGS.total_variation_weight)

            # Adding scalar summaries to the tensorboard.
            for key, value in loss_dict.items():
                tf.summary.scalar(key, value)

            # Adding Image summaries to the tensorboard.
            tf.summary.image('image/0_content_inputs', content_inputs_, 3)
            tf.summary.image('image/1_style_inputs_orig', style_inputs_orig_, 3)
            tf.summary.image('image/2_style_inputs_aug', style_inputs_, 3)
            tf.summary.image('image/3_stylized_images', stylized_images, 3)

            # Set up training
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            train_op = slim.learning.create_train_op(
                    total_loss,
                    optimizer,
                    clip_gradient_norm=FLAGS.clip_gradient_norm,
                    summarize_gradients=False)

            if tf.gfile.IsDirectory(FLAGS.vgg_checkpoint):
                checkpoint = tf.train.latest_checkpoint(FLAGS.vgg_checkpoint)
            else:
                checkpoint = FLAGS.vgg_checkpoint
                tf.logging.info('loading latest checkpoint file: {}'.format(checkpoint))
            
            init_fn = slim.assign_from_checkpoint_fn(checkpoint, slim.get_variables_to_restore())
            # sess.run([tf.local_variables_initializer()])
            # init_fn(sess)

            # Run training
            slim.learning.train(
                    train_op=train_op,
                    logdir=os.path.expanduser(FLAGS.train_dir),
                    master=FLAGS.master,
                    is_chief=FLAGS.task == 0,
                    number_of_steps=FLAGS.train_steps,
                    init_fn=init_fn,
                    save_summaries_secs=FLAGS.save_summaries_secs,
                    save_interval_secs=FLAGS.save_interval_secs)


        
if __name__ == "__main__":
    tf.disable_v2_behavior()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    tf.app.run(new_main, )
    
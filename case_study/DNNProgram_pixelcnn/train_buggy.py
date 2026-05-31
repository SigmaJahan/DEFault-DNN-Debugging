import os
import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
from model import PixelCNN  # `bits_per_dim_loss` is missing
from CustomCallback import EnhancedLoggingCallback

tfk = tf.keras
AUTOTUNE = tf.data.AUTOTUNE

def main(model_name):
    try:
        # Parsing parameters
        parser = argparse.ArgumentParser()
        parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of training epochs') # Fault 3: Hyperparameter: Epoch is too low
        parser.add_argument('-b', '--batch', type=int, default=64, help='Training batch size')
        parser.add_argument('-bf', '--buffer', type=int, default=1024, help='Buffer size for shuffling')
        parser.add_argument('-d', '--dataset', type=str, default='mnist', help='Dataset: cifar10 or mnist')
        parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning rate')
        parser.add_argument('-dc', '--lr_decay', type=float, default=0.999995, help='Learning rate decay')
        parser.add_argument('-hd', '--hidden_dim', type=int, default=32, help='Hidden dimension per channel') # Fault 2: Layer: Hidden dimension is too low
        parser.add_argument('-n', '--n_res', type=int, default=3, help='Number of res blocks') # Fault 2: Layer: Number of residual blocks is too low
        args = parser.parse_args()

        # Training parameters
        EPOCHS = args.epochs
        BATCH_SIZE = args.batch
        BUFFER_SIZE = args.buffer
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        dataset, info = tfds.load(args.dataset, with_info=True)
        train_ds, test_ds = dataset['train'], dataset['test']

        def prepare(element):
            image = element['image']
            image = tf.cast(image, tf.float32)
            return image

        def duplicate(element):
            return element, element

        train_size = info.splits['train'].num_examples
        steps_per_epoch = train_size // BATCH_SIZE
        train_ds = (train_ds.shuffle(BUFFER_SIZE)
                    .take(BUFFER_SIZE)
                    .cache()
                    .batch(BATCH_SIZE)
                    .map(prepare, num_parallel_calls=AUTOTUNE)
                    .map(duplicate)
                    .prefetch(AUTOTUNE)
                    .with_options(options))

        test_ds = (test_ds.batch(BATCH_SIZE)
                   .map(prepare, num_parallel_calls=AUTOTUNE)
                   .map(duplicate)
                   .prefetch(AUTOTUNE)
                   .with_options(options))

        model = PixelCNN(hidden_dim=args.hidden_dim, n_res=args.n_res)
        loss = tfk.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer='adam', loss=loss)  # Fault 1: `loss` is not `bits_per_dim_loss`
        callback_filename = model_name + ".csv"
        enhancedLoggingCallback = EnhancedLoggingCallback(train_ds, callback_filename)

        model.fit(
            train_ds, 
            validation_data=test_ds, 
            epochs=EPOCHS, 
            steps_per_epoch=steps_per_epoch,
            callbacks=[enhancedLoggingCallback]
        )

        model_location = os.path.join('trained_models', model_name)
        if not os.path.exists('trained_models'):
            os.makedirs('trained_models')
        model.save(model_location)
        model.summary()
        score = model.evaluate(test_ds)
        return score  

    except Exception as e:
        print(f"An error occurred: {e}")
        return 0

if __name__ == '__main__':
    model_name = "pixelcnn_buggy"
    main(model_name)
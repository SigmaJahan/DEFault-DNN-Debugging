import csv
import os
import numpy as np
import tensorflow as tf
from keras.callbacks import Callback
import gc
import psutil
 
class EnhancedLoggingCallback(Callback):
    def __init__(self, train_dataset, filename, large_weight_threshold=10.0, acc_gap_threshold=0.1, oscillation_threshold=0.01, low_accuracy_threshold=0.5, max_threshold=5, min_threshold=-5, threshold_layer=0.5, relu_threshold=0.0, layer_threshold=0.7):
        super().__init__()
        self.train_dataset = train_dataset
        self.filename = filename
        self.large_weight_threshold = large_weight_threshold
        self.acc_gap_threshold = acc_gap_threshold
        self.oscillation_threshold = oscillation_threshold
        self.low_accuracy_threshold = low_accuracy_threshold
        self.max_threshold = max_threshold
        self.min_threshold = min_threshold
        self.threshold_layer = threshold_layer
        self.relu_threshold = relu_threshold
        self.layer_threshold = layer_threshold
        self.last_metrics = {}
        self.epoch_hvp = []
        self.loss_history = []
 
        with open(self.filename, 'w', newline='') as file:
            writer = csv.writer(file)
            headers = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'large_weight_count',
                       'acc_gap_too_big', 'loss_oscillation', 'dying_relu',
                       'gradient_vanish', 'gradient_explode', 'decrease_acc_count', 'increase_loss_count',
                       'cons_mean_weight_count', 'cons_std_weight_count',
                       'nan_weight_count', 'nan_gradients_count', 'saturated_activation',
                       'mean_gradient', 'gradient_std', 'gradient_max', 'gradient_min',
                       'gradient_median', 'adjusted_lr', 'mean_activation', 'std_activation',
                       'mean_grad', 'std_grad', 'cpu_utilization', 'gpu_memory_utilization', 'memory_usage']
            writer.writerow(headers)
 
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_gradients = []
 
    @staticmethod
    def extract_accuracy(logs):
        accuracy_keys = ['accuracy', 'acc']
        mape_keys = ['mean_absolute_percentage_error']
 
        def get_first_available(log_dict, keys, default=np.nan):
            for key in keys:
                if key in log_dict:
                    return log_dict[key]
            return default
 
        train_acc = get_first_available(logs, [key for key in accuracy_keys])
        val_acc = get_first_available(logs, ['val_' + key for key in accuracy_keys])
 
        if np.isnan(train_acc):
            train_mape = get_first_available(logs, mape_keys)
            val_mape = get_first_available(logs, ['val_' + key for key in mape_keys])
            train_acc = 1 - train_mape / 100
            val_acc = 1 - val_mape / 100
 
        return train_acc, val_acc
 
    def on_train_batch_end(self, batch, logs=None):
        dataset_gradients = []
        gradient_dataset = self.train_dataset.shuffle(1000).take(3)
        for x_batch, y_batch in gradient_dataset:
            with tf.GradientTape() as tape:
                predictions = self.model(x_batch, training=True)
                loss = self.model.compiled_loss(y_batch, predictions)
            gradients = tape.gradient(loss, self.model.trainable_weights)
            dataset_gradients.append(gradients)
        average_gradients = [tf.reduce_mean(tf.stack([g[i] for g in dataset_gradients]), axis=0) for i in range(len(self.model.trainable_weights))]
        self.epoch_gradients.append(average_gradients)
 
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_loss = logs.get('loss', np.nan)
        val_loss = logs.get('val_loss', np.nan)
        train_acc, val_acc = self.extract_accuracy(logs)        
 
        weights = np.concatenate([w.flatten() for w in self.model.get_weights() if w.size > 0])
        large_weight_count = (np.abs(weights) > self.large_weight_threshold).sum()
        decrease_acc_count = 0 if self.last_metrics.get('accuracy', train_acc) > train_acc else 1
        increase_loss_count = 0 if self.last_metrics.get('loss', train_loss) < train_loss else 1
        mean_weight = np.mean(weights)
        std_weight = np.std(weights)
        nan_weight_count = np.isnan(weights).sum()
        nan_gradients_count = sum([np.isnan(g.numpy()).sum() for g in self.epoch_gradients[0] if g is not None])
        cons_mean_weight_count = 1 if self.last_metrics.get('mean_weight', mean_weight) == mean_weight else 0
        cons_std_weight_count = 1 if self.last_metrics.get('std_weight', std_weight) == std_weight else 0
        acc_gap_too_big = 1 if abs(train_acc - val_acc) > self.acc_gap_threshold else 0
        self.loss_history.append(train_loss)
        loss_oscillation = 1 if len(self.loss_history) > 1 and np.abs(self.loss_history[-2] - train_loss) > self.oscillation_threshold else 0
        dying_relu = self.check_dying_relu()
        gradient_vanish = self.check_gradient_vanish()
        gradient_explode = self.check_exploding_gradient()
        saturated_activation = self.check_saturated_activation()
 
        average_gradient = [tf.reduce_mean(tf.stack([g[i] for g in self.epoch_gradients]), axis=0) for i in range(len(self.model.trainable_weights))]
        gradient_norms = [tf.norm(g).numpy() for g in average_gradient]
        mean_gradient = np.mean(gradient_norms)
        gradient_std = np.std(gradient_norms)
        gradient_max = np.max(gradient_norms)
        gradient_min = np.min(gradient_norms)
        gradient_median = np.median(gradient_norms)
 
        adjusted_lr = self.adjust_learning_rate(epoch, logs)
        mean_activation, std_activation = self.log_activation_statistics()
        mean_grad, std_grad = self.log_gradient_statistics()
        cpu_utilization, gpu_memory_utilization = self.log_hardware_utilization()
        memory_usage = self.log_memory_usage()
 
        with open(self.filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, large_weight_count, acc_gap_too_big, loss_oscillation,
                 dying_relu, gradient_vanish, gradient_explode, decrease_acc_count, increase_loss_count, cons_mean_weight_count,
                 cons_std_weight_count, nan_weight_count, nan_gradients_count, saturated_activation, mean_gradient, gradient_std, gradient_max, gradient_min,
                 gradient_median, adjusted_lr, mean_activation, std_activation, mean_grad, std_grad,
                 cpu_utilization, gpu_memory_utilization, memory_usage])
 
        gc.collect()
 
    def adjust_learning_rate(self, epoch, logs=None):
        current_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        new_lr = current_lr * 0.9
        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
        print(f"\nEpoch {epoch+1}: Learning rate adjusted from {current_lr} to {new_lr}")
        return new_lr
 
    def log_activation_statistics(self):
        mean_activation = []
        std_activation = []
        for layer in self.model.layers:
            if hasattr(layer, 'activation'):
                outputs = layer.output
                output_func = tf.keras.backend.function([self.model.input], [outputs])
                activations = output_func(next(iter(self.train_dataset))[0])
                mean_activation.append(np.mean(activations))
                std_activation.append(np.std(activations))
        mean_activation = np.mean(mean_activation)
        std_activation = np.mean(std_activation)
        print(f"Mean Activation = {mean_activation}, Std Activation = {std_activation}")
        return mean_activation, std_activation
 
    def log_gradient_statistics(self):
        mean_grad = []
        std_grad = []
        for i, layer in enumerate(self.model.trainable_weights):
            grads = self.epoch_gradients[-1][i].numpy().flatten()
            mean_grad.append(np.mean(grads))
            std_grad.append(np.std(grads))
        mean_grad = np.mean(mean_grad)
        std_grad = np.mean(std_grad)
        print(f"Mean Gradient = {mean_grad}, Std Gradient = {std_grad}")
        return mean_grad, std_grad
 
    def log_hardware_utilization(self):
        cpu_utilization = psutil.cpu_percent()
        print(f"CPU Utilization: {cpu_utilization}%")
 
        gpu_memory_utilization = 0
        if tf.config.list_physical_devices('GPU'):
            gpu_utilization = tf.config.experimental.get_memory_info('GPU:0')
            gpu_memory_utilization = gpu_utilization['peak'] / 1024 ** 2
            print(f"GPU Memory Utilization: {gpu_memory_utilization} MB")
        return cpu_utilization, gpu_memory_utilization
 
    def log_memory_usage(self):
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.percent
        print(f"Memory Usage: {memory_usage}% used")
        return memory_usage
 
    def check_dying_relu(self):
        dead_relu_count = 0
        total_relu_count = 0
 
        for layer in self.model.layers:
            config = layer.get_config()
            if 'activation' in config and config['activation'] == 'relu':
                outputs = layer.output
                output_func = tf.keras.backend.function([self.model.input], [outputs])
                for x_batch, _ in self.train_dataset.take(1):
                    relu_outputs = output_func([x_batch])[0]
                    dead_relu_count += tf.reduce_sum(tf.cast(tf.less_equal(relu_outputs, self.relu_threshold), tf.float32))
                    total_relu_count += tf.size(relu_outputs)
 
        return (int(dead_relu_count) / total_relu_count) > self.layer_threshold if total_relu_count > 0 else 0
 
    def check_gradient_vanish(self):
        if not self.epoch_gradients:
            return 0
        flat_grads = np.concatenate([np.abs(g.numpy().flatten()) for g in self.epoch_gradients[0] if g is not None])
        if np.mean(flat_grads) < 1e-4:
            return 1
        return 0
 
    def check_exploding_gradient(self):
        if not self.epoch_gradients:
            return 0
        flat_grads = np.concatenate([np.abs(g.numpy().flatten()) for g in self.epoch_gradients[0] if g is not None])
        if np.max(flat_grads) > 70:
            return 1
        return 0
 
    def check_saturated_activation(self):
        saturated_count = 0
        total_count = 0
        inputs = self.train_dataset.shuffle(1000).take(1)
        for layer in self.model.layers:
            config = layer.get_config()
            if 'activation' in config and config['activation'] in ['sigmoid', 'tanh']:
                layer_output = layer.output
                output_func = tf.keras.backend.function([self.model.input], [layer_output])
                x_sample = next(iter(inputs))[0]
                layer_outputs = output_func(x_sample)[0]
 
                saturated = (layer_outputs >= self.max_threshold) | (layer_outputs <= self.min_threshold)
                saturated_count += tf.reduce_sum(tf.cast(saturated, tf.float32)).numpy()
                total_count += tf.size(layer_outputs).numpy()
 
        return (saturated_count / total_count) > self.threshold_layer if total_count > 0 else False
 
    def on_train_end(self, logs=None):
        if hasattr(self, 'csv_file') and self.csv_file:
            self.csv_file.close()
 
        gc.collect()
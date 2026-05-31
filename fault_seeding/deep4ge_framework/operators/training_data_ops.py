"""
Training data mutation operators.

TCL - Change Label
TRD - Delete Training Data
TUD - Unbalance Training Data
TAN - Add Noise to Training Data
TCO - Make Output Classes Overlap

These operators mutate training data arrays (x_train, y_train) before training.
They are injected into the program AST as calls before model.fit().

Note: numpy is imported lazily inside methods to allow the module to be
imported without numpy installed (needed for manifest/validation tools).
"""

import ast
import random

from .base import MutationOperator


class ChangeLabel(MutationOperator):
    """TCL: Replaces labels of the most frequent class with random other labels.

    For classification: picks the most frequent label, selects a percentage of
    its instances, and replaces their labels with random alternatives.
    """

    code = "TCL"
    name = "change_label"
    category = "Training_Data"
    applicable_to = ["FNN", "RNN", "CNN"]

    @staticmethod
    def apply_to_data(y_train, percentage=10):
        """Mutate labels. Returns modified y_train."""
        import numpy as np

        unique_labels, inverse, counts = np.unique(
            y_train, return_inverse=True, return_counts=True, axis=0
        )
        majority_idx = np.argmax(counts)
        label_indices = np.where(inverse == majority_idx)[0]

        n_to_mutate = max(1, int(len(label_indices) * percentage / 100))
        selected = np.random.choice(label_indices, size=n_to_mutate, replace=False)

        y_mutated = np.copy(y_train)
        replacement_labels = np.delete(unique_labels, majority_idx, axis=0)

        for idx in selected:
            y_mutated[idx] = replacement_labels[np.random.randint(len(replacement_labels))]

        return y_mutated


class DeleteTrainingData(MutationOperator):
    """TRD: Deletes a balanced percentage of training data across all classes."""

    code = "TRD"
    name = "delete_training_data"
    category = "Training_Data"
    applicable_to = ["FNN", "RNN", "CNN"]

    @staticmethod
    def apply_to_data(x_train, y_train, percentage=10):
        """Delete a balanced portion of training data. Returns (x_train, y_train)."""
        import numpy as np

        unique_labels, inverse, counts = np.unique(
            y_train, return_inverse=True, return_counts=True, axis=0
        )

        indices_to_keep = []
        for label_idx in range(len(unique_labels)):
            label_indices = np.where(inverse == label_idx)[0]
            n_to_delete = max(1, int(len(label_indices) * percentage / 100))
            keep = np.random.choice(
                label_indices, size=len(label_indices) - n_to_delete, replace=False
            )
            indices_to_keep.extend(keep)

        indices_to_keep = sorted(indices_to_keep)
        return x_train[indices_to_keep], y_train[indices_to_keep]


class UnbalanceTrainingData(MutationOperator):
    """TUD: Unbalances training data by deleting more from above-average classes."""

    code = "TUD"
    name = "unbalance_train_data"
    category = "Training_Data"
    applicable_to = ["FNN", "RNN", "CNN"]

    @staticmethod
    def apply_to_data(x_train, y_train, percentage=10):
        """Delete from classes with above-average counts. Returns (x_train, y_train)."""
        import numpy as np

        unique_labels, inverse, counts = np.unique(
            y_train, return_inverse=True, return_counts=True, axis=0
        )
        avg_count = np.mean(counts)

        indices_to_keep = []
        for label_idx in range(len(unique_labels)):
            label_indices = np.where(inverse == label_idx)[0]
            if counts[label_idx] > avg_count:
                n_to_delete = max(1, int(len(label_indices) * percentage / 100))
                keep = np.random.choice(
                    label_indices, size=len(label_indices) - n_to_delete, replace=False
                )
            else:
                keep = label_indices
            indices_to_keep.extend(keep)

        indices_to_keep = sorted(indices_to_keep)
        return x_train[indices_to_keep], y_train[indices_to_keep]


class AddNoise(MutationOperator):
    """TAN: Adds Gaussian noise to a percentage of training input samples.

    The noise magnitude is proportional to the standard deviation of each sample.
    """

    code = "TAN"
    name = "add_noise"
    category = "Training_Data"
    applicable_to = ["FNN", "RNN", "CNN"]

    @staticmethod
    def apply_to_data(x_train, percentage=10):
        """Add Gaussian noise to a subset of x_train. Returns modified x_train."""
        import numpy as np

        x_noisy = np.copy(x_train)
        n_samples = len(x_train)
        n_to_mutate = max(1, int(n_samples * percentage / 100))
        selected = np.random.choice(n_samples, size=n_to_mutate, replace=False)

        for idx in selected:
            sample = x_train[idx]
            sigma = np.std(sample.flatten())
            noise = np.random.normal(0, sigma * percentage / 100, sample.shape)
            x_noisy[idx] = sample + noise

        return x_noisy


class MakeOutputClassesOverlap(MutationOperator):
    """TCO: Duplicates samples from one class and labels them as another class.

    Picks the two most frequent classes, duplicates a percentage of class-1
    samples, and appends them with class-2 labels.
    """

    code = "TCO"
    name = "make_output_classes_overlap"
    category = "Training_Data"
    applicable_to = ["FNN", "RNN", "CNN"]

    @staticmethod
    def apply_to_data(x_train, y_train, percentage=10):
        """Create class overlap. Returns (x_train, y_train)."""
        import numpy as np

        unique_labels, inverse, counts = np.unique(
            y_train, return_inverse=True, return_counts=True, axis=0
        )

        sorted_indices = np.argsort(-counts)
        idx1, idx2 = sorted_indices[0], sorted_indices[1]
        label2 = unique_labels[idx2]

        label1_indices = np.where(inverse == idx1)[0]
        n_to_duplicate = max(1, int(len(label1_indices) * percentage / 100))
        selected = np.random.choice(label1_indices, size=n_to_duplicate, replace=False)

        x_duplicated = x_train[selected]
        y_duplicated = np.full((n_to_duplicate,) + y_train.shape[1:], label2)

        return np.concatenate([x_train, x_duplicated]), np.concatenate([y_train, y_duplicated])

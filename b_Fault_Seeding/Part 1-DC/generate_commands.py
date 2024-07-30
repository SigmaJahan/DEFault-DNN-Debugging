import itertools
import os

files = [file for file in os.listdir('test_models') if file.endswith('.py')]

mutation_operators = ['change_batch_size', 'change_epochs', 'change_learning_rate', 'disable_batching', 'change_activation_function', 'remove_activation_function', 'add_activation_function', 'change_weights_initialisation', 'change_optimisation_function', 'change_gradient_clip', 'change_earlystopping_patience', 'add_bias', 'remove_bias', 'change_loss_function', 'change_dropout_rate', 'add_weights_regularisation', 'change_weights_regularisation', 'remove_weights_regularisation']
combinations = itertools.product(files, mutation_operators)

with open('run_combinations.sh', 'w') as file:
    file.write('#!/bin/bash\n\n')
    for combination in combinations:
        command = f"python run_deepcrime.py {combination[0]} {combination[1]}\n"
        file.write(command)

os.chmod('run_combinations.sh', 0o755)

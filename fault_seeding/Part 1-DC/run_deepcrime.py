import os
import shutil
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import importlib
import sys
import utils.properties as props
import utils.constants as const
import run_deepcrime_properties as dc_props
from deep_crime import mutate as run_deepcrime_tool
from utils.constants import save_paths
from mutation_score import calculate_dc_ms

data = {
    'subject_name': '',
    'subject_path': '',
    'root': os.path.dirname(os.path.abspath(__file__)),
    'mutations': [],
    'mode': 'test'
}

def run_automate(file, mutation):
    data['subject_name'] = 'deepmultifix' + file.split('.')[0]
    data['subject_path'] = os.path.join('test_models', file)
    data['mutations'] = [mutation]
    dc_props.write_properties(data)
    run_deepcrime_tool()
    print("Finished all, exit")


if __name__ == '__main__':
    file = sys.argv[1]
    mutation = sys.argv[2]
    run_automate(file, mutation)
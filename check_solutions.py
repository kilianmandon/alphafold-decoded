import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os

def execute_notebook(notebook_path, cwd):
    # Load the notebook
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    
    # Create a preprocessor
    executor = ExecutePreprocessor(timeout=None)
    
    # Execute the notebook
    try:
        executor.preprocess(nb, {'metadata': {'path': cwd}})
    except Exception as e:
        # If there is any error, print it
        print(f"Error executing notebook {notebook_path}: {e}")
        return False
    else:
        print(f"Notebook {notebook_path} executed successfully.")
        return True



notebook_paths = [
    'solutions/tensor_introduction/tensor_introduction.ipynb',
    'solutions/machine_learning/machine_learning.ipynb',
    'solutions/attention/attention.ipynb',
    'solutions/feature_extraction/feature_extraction.ipynb',
    'solutions/evoformer/evoformer.ipynb',
    'solutions/feature_embedding/feature_embedding.ipynb',
    'solutions/geometry/geometry.ipynb',
    'solutions/structure_module/structure_module.ipynb',
    'solutions/model/model.ipynb',
]


for path in notebook_paths:
    name = path.split('/')[0]
    if not execute_notebook(path, 'solutions'):
        break

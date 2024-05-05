import re
import numpy as np
import os
from pathlib import Path
import shutil
import nbformat


def parse_file(filename, out_name):
    """Parses a Jupyter Notebook file, searches for TODO blocks and replaces them 
        with placeholder text.

    Args:
        filename: The name of the file to parse.
        out_name: The nae of the file to write to.
    """

    is_ipynb = filename.endswith('ipynb')

    with open(filename, 'r') as f:
        lines = f.readlines()

        hashtag_inds = [i for i,l in enumerate(lines) if '####################' in l]
        assert len(hashtag_inds)%4==0
        groups = [hashtag_inds[i:i+4] for i in range(0, len(hashtag_inds), 4)]
        to_replace_inds = []

        for group_inds in groups:
            todo_lines = '\n'.join(lines[group_inds[0]:group_inds[1]])
            end_lines = '\n'.join(lines[group_inds[2]:group_inds[3]])
            assert 'TODO' in todo_lines
            assert 'END OF YOUR CODE' in end_lines
            to_replace_inds.append((group_inds[1]+1, group_inds[2]))

        def gen_replace_with(leading_whitespace_code, leading_whitespace_ipynb):
            if is_ipynb:
                basic_line_content = [
                    '# Replace \\"pass\\" statement with your code',
                    'pass',
                ]
            else:
                basic_line_content = [
                    '# Replace "pass" statement with your code',
                    'pass'
                ]

            if is_ipynb:
                basic = ' '*leading_whitespace_ipynb + '"\\n",\n'
            else:
                basic = '\n'

            to_return = [basic]
            for content in basic_line_content:
                if is_ipynb:
                    line = ' '*leading_whitespace_ipynb + '"' + ' '*leading_whitespace_code + f'{content}\\n",\n'
                else:
                    line = ' '*leading_whitespace_code + f'{content}\n'

                to_return.append(line)
            to_return += [basic]

            return to_return

        def calculate_leading_whitespace(line):
            if is_ipynb:
                quotes = [i for i,c in enumerate(line) if c=='"']
                real_line = line[quotes[0]+1:quotes[-1]]
            else:
                real_line = line

            unescaped = real_line.replace('\\n', '\n')
            if not any(char.isalnum() for char in unescaped):
                return -1

            return len(real_line) - len(real_line.lstrip())
            

        for replace_start, replace_stop in reversed(to_replace_inds):
            leading_whitespace_code = np.array([calculate_leading_whitespace(a) for a in lines[replace_start:replace_stop]])
            leading_whitespace_code = leading_whitespace_code[leading_whitespace_code != -1]
            if leading_whitespace_code.size == 0:
                leading_whitespace_code = 0
            else:
                leading_whitespace_code = np.min(leading_whitespace_code)
            leading_whitespace_ipynb = np.array([len(l)-len(l.lstrip()) for l in lines[replace_start:replace_stop]])
            leading_whitespace_ipynb = np.max(leading_whitespace_ipynb)
            lines = lines[:replace_start] + gen_replace_with(leading_whitespace_code, leading_whitespace_ipynb)  + lines[replace_stop:]

        Path(out_name).parent.mkdir(parents=True, exist_ok=True)

        with open(out_name, 'w') as f:
            f.writelines(lines)


python_paths = [
    'tensor_introduction/tensor_introduction.ipynb',
    'machine_learning_introduction/machine_learning_introduction.ipynb',
    'machine_learning_introduction/feed_forward.py',
    'attention/attention.ipynb',
    'attention/mha.py',
    'feature_extraction/feature_extraction.py',
    'feature_extraction/feature_extraction.ipynb',
    'evoformer/dropout.py',
    'evoformer/msa_stack.py',
    'evoformer/pair_stack.py',
    'evoformer/evoformer.py',
    'evoformer/evoformer.ipynb',
    'feature_embedding/feature_embedding.ipynb',
    'feature_embedding/extra_msa_stack.py',
    'feature_embedding/input_embedder.py',
    'feature_embedding/recycling_embedder.py',
    'geometry/geometry.py',
    'geometry/geometry.ipynb',
    'geometry/residue_constants.py',
    'structure_module/ipa.py',
    'structure_module/structure_module.py',
    'structure_module/structure_module.ipynb',
    'model/utils.py',
    'model/model.py',
    'model/model.ipynb',
]

file_copy_paths = [
    'feature_extraction/alignment_tautomerase.a3m',
    'model/download_openfold_params.sh',
]

folder_copy_paths = [
    'tensor_introduction/control_values',
    'machine_learning_introduction/control_values',
    'attention/control_values',
    'feature_extraction/control_values',
    'evoformer/control_values',
    'evoformer/images',
    'feature_embedding/control_values',
    'feature_embedding/images',
    'geometry/control_values',
    'structure_module/control_values',
    'model/control_values',
]

import glob

# Pattern 1: All __init__.py files directly within a subfolder of tutorials
pattern1 = "solutions/*/__init__.py"
files1 = glob.glob(pattern1)

# Pattern 2: All __init__.py files in a 'control_values' subfolder within tutorials
pattern2 = "solutions/*/control_values/__init__.py"
files2 = glob.glob(pattern2)

# Combine the results 
all_init_files = files1 + files2
all_init_files = [path.replace('solutions/', '') for path in all_init_files]
file_copy_paths += all_init_files

def delete_tutorials_contents():
    tutorials_path = 'tutorials'
    if os.path.exists(tutorials_path):
        shutil.rmtree(tutorials_path)
        print("The 'tutorials' folder and its contents have been deleted.")
    else:
        print("The 'tutorials' folder doesn't exist.")

delete_tutorials_contents()



def clear_notebook_outputs(notebook_path):
    nb = nbformat.read(notebook_path, as_version=4)
    nb.metadata.clear()  # Clear metadata first (important for outputs)
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.outputs = []      
    nbformat.write(nb, notebook_path)


# --- Assertions ---
for path in python_paths + file_copy_paths:
    full_source = f'solutions/{path}'
    full_target = f'tutorials/{path}' 
    assert os.path.isfile(full_source), f"Source file not found: {full_source}"

for path in folder_copy_paths:
    full_source = f'solutions/{path}'
    full_target = f'tutorials/{path}' 
    assert os.path.isdir(full_source), f"Source folder not found: {full_source}"

# --- Copying ---
for path in python_paths:
    full_source = f'solutions/{path}'
    full_target = f'tutorials/{path}' 
    parse_file(full_source, full_target)

    if full_target.endswith('.ipynb'):
        clear_notebook_outputs(full_target)
        clear_notebook_outputs(full_source)

for path in folder_copy_paths:
    full_source = f'solutions/{path}'
    full_target = f'tutorials/{path}' 
    shutil.copytree(full_source, full_target)

for path in file_copy_paths:
    full_source = f'solutions/{path}'
    full_target = f'tutorials/{path}' 
    shutil.copyfile(full_source, full_target)

def remove_overwrite_results_option(filename):
    with open(filename, 'r') as f:
        data = f.read()
    data = re.sub(r"\s*,\s*overwrite_results\s*=\s*False\s*", "", data)
    data = re.sub(r"\s*,\s*overwrite_results\s*=\s*overwrite_results\s*", "", data)
    pattern = r"^\s*if\s+overwrite_results:\s*(?s:.*?)\n\s*\n"
    data = re.sub(pattern, "", data, flags=re.MULTILINE)
    with open(filename, 'w') as f:
        f.write(data)


control_save_to_remove = [
    'tutorials/evoformer/control_values/evoformer_checks.py', 
    'tutorials/feature_embedding/control_values/embedding_checks.py',
    'tutorials/structure_module/control_values/structure_module_checks.py',
    'tutorials/model/control_values/model_checks.py',
]
for control_save_remove in control_save_to_remove:
    remove_overwrite_results_option(control_save_remove)

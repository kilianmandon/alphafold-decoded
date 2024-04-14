import re
import numpy as np

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

            return len(real_line) - len(real_line.lstrip())
            

        for replace_start, replace_stop in reversed(to_replace_inds):
            leading_whitespace_code = np.array([calculate_leading_whitespace(a) for a in lines[replace_start:replace_stop]])
            leading_whitespace_code = leading_whitespace_code[leading_whitespace_code != 1]
            if leading_whitespace_code.size == 0:
                leading_whitespace_code = 0
            else:
                leading_whitespace_code = np.min(leading_whitespace_code)
            leading_whitespace_ipynb = np.array([len(l)-len(l.lstrip()) for l in lines[replace_start:replace_stop]])
            leading_whitespace_ipynb = np.max(leading_whitespace_ipynb)
            lines = lines[:replace_start] + gen_replace_with(leading_whitespace_code, leading_whitespace_ipynb)  + lines[replace_stop:]

        with open(out_name, 'w') as f:
            f.writelines(lines)


            
        




    


# Example usage
inp_name = 'solutions/attention/attention.ipynb'
out_name = 'tutorials/attention/attention.ipynb'
parse_file(inp_name, out_name)

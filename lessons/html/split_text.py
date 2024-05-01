import re
from pathlib import Path


def is_valid_html(html_string):
    """Checks if a string is valid HTML with correctly closed tags.

    Args:
        html_string (str): The HTML string to validate.

    Returns:
        bool: True if the HTML is valid, False otherwise.
    """

    # Tags that must be closed
    p_open = [m.start() for m in re.finditer('<p', html_string)]
    p_close = [m.start() for m in re.finditer('</p>', html_string)]
    stack = []
    while p_open or p_close:
        if p_open and (not p_close or p_open[0] < p_close[0]):
            stack.append('')
            p_open.pop(0)
        else:
            if len(stack) == 0:
                return False
            else:
                stack.pop(0)
                p_close.pop(0)

    return len(stack) == 0 and len(p_open) == 0 and len(p_close) == 0

def group_lines(lines):
    all_groups = []
    current_group = []
    for line in lines:
        current_group.append(line)
        if is_valid_html(''.join(current_group)):
            all_groups.append(''.join(current_group))
            current_group = []

    return all_groups


def split_text_by_characters(file_path, max_chunk_size=9000):
    """Splits a file into chunks based on character count, respecting lines.

    Args:

        file_path (str): The path to your input file.

        max_chunk_size (int): The maximum number of characters per chunk.

    """

    chunks = []

    current_chunk = []

    current_chunk_size = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        segments = group_lines(lines)

    for segment in segments:

        if current_chunk_size + len(segment) <= max_chunk_size:

            current_chunk.append(segment)

            current_chunk_size += len(segment)

        else:

            # Join segments into a single string
            chunks.append("".join(current_chunk))

            current_chunk = [segment]  # Start new chunk with current segment

            current_chunk_size = len(segment)

    # Add the last chunk

    if current_chunk:  # Check if the last chunk isn't empty

        chunks.append("".join(current_chunk))

    # Save chunks to files

    for i, chunk in enumerate(chunks):

        output_path = Path(file_path).parents[0] / f"chunk_{i+1}.html"
        print(f'Is chunk {i} valid html? {is_valid_html(chunk)}')

        with open(output_path, 'w', encoding='utf-8') as out_file:

            out_file.write(chunk)


# Example usage

# Replace with your file path
file_path = "lessons/html/tensor_introduction.html_css_classes.html"

split_text_by_characters(file_path)

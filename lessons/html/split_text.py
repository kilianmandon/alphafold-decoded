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
    # p_open = [m.start() for m in re.finditer(r'<p[^/>]*>', html_string)]
    # p_close = [m.start() for m in re.finditer('</p>', html_string)]
    tags_to_check = ['p', 'section']
    for tag in tags_to_check:
        tag_open, tag_close = find_html_tags(html_string, tag)

        stack = []
        while tag_open or tag_close:
            if tag_open and (not tag_close or tag_open[0] < tag_close[0]):
                stack.append('')
                tag_open.pop(0)
            else:
                if len(stack) == 0:
                    return False
                else:
                    stack.pop(0)
                    tag_close.pop(0)
        if len(stack) != 0 or len(tag_open) != 0 or len(tag_close) != 0:
            return False

    return True

def find_html_tags(text, tag):
    tag_open = [m.start() for m in re.finditer(f'<{tag}[^/>]*>', text)]
    tag_close = [m.start() for m in re.finditer(f'</{tag}>', text)]
    return tag_open, tag_close
    

def group_lines(lines):
    all_groups = []
    current_group = []
    for line in lines:
        current_group.append(line)
        if is_valid_html(''.join(current_group)):
            all_groups.append(''.join(current_group))
            current_group = []

    if current_group:
        rest = ''.join(current_group)
        print(f'Overhanging group of length {len(rest)}')
        p_open = [m.start() for m in re.finditer(r'<p[^/>]*>', rest)]
        p_close = [m.start() for m in re.finditer('</p>', rest)]
        print(len(p_open), len(p_close))
        with open('lessons/html/problematic_group.txt', 'w') as f:
            f.write(rest)

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

    print(f'Number of chunks: {len(chunks)}')
    for i, chunk in enumerate(chunks):

        output_path = Path(file_path).parents[0] / f"chunk_{i+1}.html"
        print(f'Is chunk {i} valid html? {is_valid_html(chunk)}')

        with open(output_path, 'w', encoding='utf-8') as out_file:

            out_file.write(chunk)


# Example usage

# Replace with your file path
file_path = "lessons/html/machine_learning_css_classes.html"

split_text_by_characters(file_path)

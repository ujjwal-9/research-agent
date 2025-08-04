import argparse
import os

import markdown
import pdfkit


def render_markdown_to_pdf(markdown_content, output_file):
    """
    Converts Markdown content to a PDF file.

    Args:
        markdown_content (str): The Markdown content as a string.
        output_file (str): Path to the output PDF file.

    Returns:
        None
    """
    # Convert Markdown to HTML
    html_content = markdown.markdown(markdown_content)

    # Generate a temporary HTML file
    temp_html = "temp_output.html"
    with open(temp_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    try:
        # Convert HTML to PDF
        pdfkit.from_file(temp_html, output_file)
        print(f"PDF successfully created: {output_file}")
    except Exception as e:
        print(f"Error generating PDF: {e}")
    finally:
        # Cleanup temporary HTML file
        if os.path.exists(temp_html):
            os.remove(temp_html)


def list_files_in_folder(folder_path):
    """
    List all files in the given folder and its subfolders.

    Args:
        folder_path (str): Path to the folder to scan.

    Returns:
        list: A list of file paths.
    """
    files_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            files_list.append(os.path.join(root, file))
    return files_list


def validate_files(input_paths):
    """
    Validate if the input paths exist and are either files or folders.

    Args:
        input_paths (list): List of input paths.

    Returns:
        list: A list of valid file paths.
    """
    valid_files = []
    for path in input_paths:
        if os.path.isfile(path):
            valid_files.append(path)
        elif os.path.isdir(path):
            valid_files.extend(list_files_in_folder(path))
        else:
            print(f"Warning: '{path}' is not a valid file or directory.")
    return valid_files


def main():
    parser = argparse.ArgumentParser(
        description="Process data and schema files or folders."
    )
    parser.add_argument(
        "--data",
        nargs="+",
        type=str,
        required=True,
        help="Provide a folder path or one or more file paths for data.",
    )
    parser.add_argument(
        "--schema",
        nargs="+",
        type=str,
        required=True,
        help="Provide a folder path or one or more file paths for schema.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="Provide a file path for saving pdf.",
    )
    args = parser.parse_args()

    # Process --data input
    data_paths = args.data
    data_files = validate_files(data_paths)
    print("\nData Files:")
    if data_files:
        for file in data_files:
            print(file)
    else:
        print("No valid data files found.")

    # Process --schema input
    schema_paths = args.schema
    schema_files = validate_files(schema_paths)
    print("\nSchema Files:")
    if schema_files:
        for file in schema_files:
            print(file)
    else:
        print("No valid schema files found.")

    return data_files, schema_files, args.output

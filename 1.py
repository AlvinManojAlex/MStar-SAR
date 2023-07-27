import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os

def execute_notebook(notebook_path):
    print(f"Executing {notebook_path}...")
    with open(notebook_path) as notebook_file:
        notebook_content = nbformat.read(notebook_file, as_version=4)
        executor = ExecutePreprocessor(timeout=-1, allow_errors=True)
        executor.preprocess(notebook_content, {'metadata': {'path': os.path.dirname(notebook_path)}, 'stdout': None, 'stderr': None})
        nbformat.write(notebook_content, notebook_path)
    print(f"{notebook_path} executed successfully.")

if __name__ == "__main__":
    notebooks = [ "densenet.ipynb","Efficientnet.ipynb", "googlenettest2.ipynb"]

    for notebook in notebooks:
        execute_notebook(notebook)

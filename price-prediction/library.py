from IPython import get_ipython
from pathlib import Path
import papermill as pm
import nbformat
import sys
import os

__original__stdout__ = sys.stdout
  
def run_notebook_without_ipython(notebook_path, suppress_output=True):
    original_value = sys.stdout
    if suppress_output:
        sys.stdout = open(notebook_path + '.outputs.txt', 'w')

    with open(notebook_path, 'r') as f:
        notebook = nbformat.read(f, as_version=4)
    local_scope = {}
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            try:
                exec(cell.source, local_scope) 
            except Exception as e:
                print(f"Error executing cell: {e}")

    if suppress_output:
        sys.stdout = original_value

    return local_scope

def inject_parameters_into_notebook(notebook_path, params, output_path):
    with open(notebook_path, 'r') as f:
        notebook = nbformat.read(f, as_version=4)
    param_cell_code = '\n'.join([f'{key} = {repr(value)}' for key, value in params.items()])
    param_cell = nbformat.v4.new_code_cell(param_cell_code)
    
    # Remove the original first cell
    if notebook.cells:
        notebook.cells.pop(0)
    notebook.cells.insert(0, param_cell)
    
    with open(output_path, 'w') as f:
        nbformat.write(notebook, f)


def getAssets(tickers, eraseExisting=False, plotOutput='get-assets_date-ranges'):
    notebook_path = 'get-assets.ipynb'
    params = {
        'tickers': tickers,
        'eraseExisting': eraseExisting,
        'plotOutput': plotOutput
    }
    output_path = str(Path(notebook_path).with_suffix('.parameterized.ipynb'))
    inject_parameters_into_notebook(notebook_path, params, output_path)
    local_scope = run_notebook_without_ipython(output_path)
    return local_scope['output_data']

if __name__ == '__main__':
    print(getAssets(['DOGEUSDC', 'ETHUSDC']))
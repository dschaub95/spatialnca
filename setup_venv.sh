uv sync --managed-python
uv pip install torch_scatter torch_cluster pyg_lib torch_sparse -f https://data.pyg.org/whl/torch-2.7.0+cu126.html
uv pip install -e .
unset VIRTUAL_ENV
export UV_LINK_MODE=symlink

uv venv --python 3.12
source .venv/bin/activate
uv sync
uv pip install torch_scatter torch_cluster pyg_lib torch_sparse -f https://data.pyg.org/whl/torch-2.7.0+cu126.html
uv pip install -e .
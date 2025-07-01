# Set UV cache directory if /epyc exists or if UV_CACHE_DIR is specified
if [ -d "/epyc" ]; then
    export UV_CACHE_DIR=/epyc/projects/dschaub/.uv-cache
elif [ ! -z "${UV_CACHE_DIR}" ]; then
    export UV_CACHE_DIR="${UV_CACHE_DIR}"
fi
echo "UV cache directory: ${UV_CACHE_DIR}"

# rm -r .venv
uv venv
source .venv/bin/activate
uv sync
uv pip install torch_scatter torch_cluster pyg_lib torch_sparse -f https://data.pyg.org/whl/torch-2.7.0+cu126.html
uv pip install -e .
build-docs:
    PYTHONPATH={{justfile_directory()}} uv run --all-extras sphinx-build -W -b html docs docs/build

serve-docs: build-docs
    uv run python -m http.server -d docs/build/

clean-docs:
    rm -rf docs/build
    rm -rf docs/api/generated
    rm -rf docs/jupyter_execute

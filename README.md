# llama2-py-inference
Pythonized version of [llama2.c](https://github.com/karpathy/llama2.c)

# Poetry environment
The dependencies and environment are managed using [Poetry](https://python-poetry.org/).

## Pre-requisites
**Important** Run first `poetry install` from the command line in order
for the modules in the _src_ directory to become accessible.

## Running llama2

`poetry run llama2 data/stories15M.bin 0.8 256 "In that small Swiss town"`

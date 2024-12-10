# Welcome to MkDocs

For full documentation visit [mkdocs.org](https://www.mkdocs.org).

## Commands

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.

## Requirements

- Python 3.11
- [Ollama](https://ollama.com/).


## Installation

- Python requirements:

```bash
pip install -r requirements.txt
```

- [Ollama](https://ollama.com/download):

```bash
ollama run qwen2.5-coder
```

## Usage

```bash
cd src
python main.py
```

You can see help message by running:

```bash
python main.py -h
```

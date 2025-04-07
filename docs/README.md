# WDBX Documentation

This directory contains the documentation for the WDBX project.

## Building the Documentation

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Installation

1. Activate your virtual environment:

```bash
# For Windows
.\.venv\Scripts\Activate.ps1

# For Linux/Mac
source .venv/bin/activate
```

2. Install the required packages:

```bash
pip install -r docs/requirements.txt
```

### Building

#### Using PowerShell (Windows)

Run the PowerShell script:

```powershell
.\docs\build_docs.ps1
```

#### Using Python (Cross-platform)

Run the Python script:

```bash
python docs/build_docs.py
```

#### Manually

Navigate to the docs directory and run Sphinx directly:

```bash
cd docs
sphinx-build -b html . _build/html
```

### Viewing the Documentation

After building, open `docs/_build/html/index.html` in your web browser.

## Documentation Structure

- `index.rst`: Main entry point for the documentation
- `conf.py`: Sphinx configuration file
- `_static/`: Static files (CSS, JavaScript, images)
- `_templates/`: Custom templates
- `*.md`, `*.rst`: Documentation content files

## Contributing to Documentation

1. Write content in Markdown (`.md`) or reStructuredText (`.rst`) format
2. Add your file to the appropriate toctree in `index.rst`
3. Build and test the documentation locally
4. Submit a pull request 
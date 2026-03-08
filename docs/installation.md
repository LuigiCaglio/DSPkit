# Installation

```bash
git clone https://github.com/LuigiCaglio/DSPkit.git
cd DSPkit
pip install -e .
```

## Requirements

| Package | Minimum version |
|---|---|
| Python | 3.10 |
| NumPy | 1.24 |
| SciPy | 1.10 |
| Matplotlib | 3.7 |

## Optional extras

### Development (testing)

```bash
pip install -e ".[dev]"
```

Installs `pytest` and `pytest-cov`.

```bash
pytest          # run all tests
pytest --cov    # with coverage report
```

### Documentation

```bash
pip install -e ".[docs]"
mkdocs serve    # live preview at http://127.0.0.1:8000
mkdocs build    # build static site into site/
```

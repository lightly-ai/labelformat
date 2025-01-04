# Installation

Installing **Labelformat** is straightforward. Follow the steps below to set up Labelformat in your development environment.

## Prerequisites

- **Python 3.7 or higher:** Ensure you have Python installed on Windows, Linux, or macOS.
- **pip:** Python's package installer. It typically comes with Python installations.

## Installation using package managers

Labelformat is available on PyPI and can be installed using various package managers:

=== "pip"
    ```bash
    pip install labelformat
    ```

=== "Poetry"
    ```bash
    poetry add labelformat
    ```

=== "Conda"
    ```bash
    conda install -c conda-forge labelformat
    ```

=== "Rye"
    ```bash
    rye add labelformat
    ```

## Installation from Source

If you prefer to install Labelformat from the source code, follow these steps:

1. Clone the Repository:
   ```bash
   git clone https://github.com/lightly-ai/labelformat.git
   cd labelformat
   ```
2. Install Dependencies:
   Labelformat uses Poetry for dependency management. Ensure you have Poetry installed:
   ```bash
   pip install poetry
   ```
3. Set Up the Development Environment:
   ```bash
   poetry install
   ```

## Updating Labelformat

To update Labelformat to the latest version, run:
```bash
pip install --upgrade labelformat
```
  

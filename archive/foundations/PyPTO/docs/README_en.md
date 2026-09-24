# Project Documentation

## Introduction

This directory provides source file information for the [PyPTO Documentation Center](https://pypto.gitcode.com), including environment setup, programming guides, API references, and more.

## Contribution

You are welcome to contribute to the documentation. For details, refer to the [Documentation Contribution Guide](./CONTRIBUTION_DOC_en.md). Follow the documentation writing specifications and submit according to the process rules. After review and approval, the content appears in this project's docs directory and the documentation center webpage. If you have any comments or suggestions about the documentation, submit them in Issues.

## Directory Description

The key directory structure is as follows:

```txt
docs/
├── zh/                         # Chinese documentation
│   ├── install                 # Environment setup
│   ├── invocation              # Sample execution
│   ├── tutorials               # PyPTO Operator Development
│   └── api                     # PyPTO APIs
├── CONTRIBUTION_DOC.md         # Documentation contribution guide
└── README.md                   # This file
```

## Documentation Build

The PyPTO programming guide and API documentation can be generated using the Sphinx tool. When a documentation PR is merged, documentation build is automatically triggered. Local build is also supported. Before building documentation locally, install the necessary modules. The following are the specific steps.

1. Download the PyPTO repository code.

   ```bash
   git clone https://gitcode.com/cann/pypto.git
   ```

2. Enter the docs directory and install the dependencies listed in `requirements.txt`.

   ```bash
   cd docs
   pip install -r requirements.txt
   ```

3. Execute the following command in the docs directory for documentation build.

   ```bash
   make html
   ```

4. After the build completes, the `_build/html` directory is created. Execute the following command to start an HTTP server to serve the documentation.

   ```bash
   cd _build/html
   python3 -m http.server 8000
   ```

   The default port is 8000. You can also specify a custom port.

5. Access `http://localhost:8000` in the browser to view the documentation.

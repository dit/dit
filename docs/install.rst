Install
=======

The easiest way to install is:

.. code-block:: bash

  pip install dit

If you want to install ``dit`` within a conda environment, you can simply do:

.. code-block:: bash

  conda install -c conda-forge dit

For development, we recommend `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: bash

  git clone https://github.com/dit/dit.git
  cd dit
  uv sync --extra dev

This installs ``dit`` in editable mode with all development dependencies
(tests, docs, linting, type checking, and optional backends).

To install specific optional extras:

.. code-block:: bash

  # JAX optimization backend
  pip install "dit[jax]"

  # PyTorch optimization backend
  pip install "dit[torch]"

  # xarray-based distributions
  pip install "dit[xarray]"

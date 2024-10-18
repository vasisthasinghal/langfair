Usage
=====

.. _installation:

.. _gettingstarted:

Installation
------------

To use LLaMBDA, we recommend you first create an isolated environment with python 3.9 (or higher):

.. code-block:: console

    conda create python=3.9 --name llambda-env


Activate the new environment before installing `llambda`.

.. code-block:: console
    
    conda activate llambda-env

The shell should now look like `(llambda-env) $`.

Install LLaMBDA using pip directly from the GitHub repository.

.. code-block:: console

    (llambda-env) $ pip install --user git+https://github.com/cvs-health/llambda.git


If you would like to access your virtual env from a notebook, install it as a Jupyter kernel:

.. code-block:: console

    python kernel install --name "llambda-env" --user
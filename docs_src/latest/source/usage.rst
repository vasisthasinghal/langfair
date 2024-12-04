Get Started
===========

.. _installation:

.. _gettingstarted:

Installation
------------

We recommend creating a new virtual environment using venv before installing LangFair. To do so, please follow instructions `here <https://docs.python.org/3/library/venv.html>`_.

.. code-block:: console

   pip install langfair

Usage Example
-------------

Below is a sample of code illustrating how to use LangFair's `AutoEval` class for text generation and summarization use cases. The below example assumes the user has already defined parameters
``DEPLOYMENT_NAME``, ``API_KEY``, ``API_BASE``, ``API_TYPE``, ``API_VERSION``, and a list of prompts from their use case `prompts`.

|
Create `langchain` LLM object.

.. code-block:: python

   from langchain_openai import AzureChatOpenAI
   # import torch # uncomment if GPU is available
   # device = torch.device("cuda") # uncomment if GPU is available

   llm = AzureChatOpenAI(
      deployment_name=DEPLOYMENT_NAME,
      openai_api_key=API_KEY,
      azure_endpoint=API_BASE,
      openai_api_type=API_TYPE,
      openai_api_version=API_VERSION,
      temperature=0.4 # User to set temperature
   )

.. note::

   You can use any of `LangChain's LLM classes <https://js.langchain.com/docs/integrations/chat/>`_ in place of AzureChatOpenAI. 
   
   Also, to avoid rate limit errors, use LangChain's `InMemoryRateLimiter <https://api.python.langchain.com/en/latest/rate_limiters/langchain_core.rate_limiters.InMemoryRateLimiter.html#>`_

|
Run the `AutoEval` method for automated bias / fairness evaluation

.. code-block:: python

   from langfair.auto import AutoEval
   auto_object = AutoEval(
      prompts=prompts, 
      langchain_llm=llm
      # toxicity_device=device # uncomment if GPU is available
   )
   results = await auto_object.evaluate() 

.. image:: ./_static/images/autoeval_process.png
   :width: 800
   :alt: AutoEval Process
   
|
Print the results and export to .txt file.

.. code-block:: python

   auto_object.export_results(file_name="metric_values.txt")
   auto_object.print_results()

.. image:: ./_static/images/autoeval_output.png
   :width: 500
   :align: center
   :alt: AutoEval Output

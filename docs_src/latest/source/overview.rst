Overview
========

LangFair: Use-Case Level LLM Bias and Fairness Assessments
-----------------------------------------------------------

LangFair is a comprehensive Python library designed for conducting bias and fairness assessments of large language model (LLM) use cases. 
This repository includes a comprehensive framework for `choosing bias and fairness metrics <https://github.com/cvs-health/langfair/tree/main#choosing-bias-and-fairness-metrics-for-an-llm-use-case>`_,
along with `demo notebooks <https://github.com/cvs-health/langfair/tree/main/examples>`_ and a `technical playbook <https://arxiv.org/abs/2407.10853>`_ that discusses LLM bias and fairness risks, evaluation metrics, and best practices. 

Why choose LangFair?
--------------------

Static benchmark assessments, which are typically assumed to be sufficiently representative, often fall short in capturing the risks associated with all possible use cases of LLMs. These models are increasingly used in various applications, including recommendation systems, classification, text generation, and summarization. However, evaluating these models without considering use-case-specific prompts can lead to misleading assessments of their performance, especially regarding bias and fairness risks.

LangFair addresses this gap by adopting a Bring Your Own Prompts (BYOP) approach, allowing users to tailor bias and fairness evaluations to their specific use cases. This ensures that the metrics computed reflect the true performance of the LLMs in real-world scenarios, where prompt-specific risks are critical. Additionally, LangFair's focus is on output-based metrics that are practical for governance audits and real-world testing, without needing access to internal model states.

Choosing Bias and Fairness Metrics for an LLM Use Case
------------------------------------------------------

Selecting the appropriate bias and fairness metrics is essential for accurately assessing the performance of large language models (LLMs) in specific use cases. Instead of attempting to compute all possible metrics, practitioners should focus on a relevant subset that aligns with their specific goals and the context of their application.

Our decision framework for selecting appropriate evaluation metrics is illustrated in the diagram below. For more details, refer to our `technical playbook <https://arxiv.org/abs/2407.10853>`_.

.. image:: ./_static/images/use_case_framework.PNG
   :width: 800
   :align: center
   :alt: Use Case Framework


.. note::

   Fairness through unawareness means none of the prompts for an LLM 
   use case include any mention of protected attribute words.

Supported Bias and Fairness Metrics 
-----------------------------------

Bias and fairness metrics offered by LangFair are grouped into several categories. The full suite of metrics is displayed below.

**Toxicity Metrics**

* Expected Maximum Toxicity `[Gehman et al., 2020] <https://arxiv.org/abs/2009.11462>`_
* Toxicity Probability `[Gehman et al., 2020] <https://arxiv.org/abs/2009.11462>`_
* Toxic Fraction `[Liang et al., 2023] <https://arxiv.org/abs/2211.09110>`_

**Counterfactual Fairness Metrics**

* Strict Counterfactual Sentiment Parity `[Huang et al., 2020] <https://arxiv.org/abs/1911.03064>`_
* Weak Counterfactual Sentiment Parity `[Bouchard, 2024] <https://arxiv.org/abs/2407.10853>`_
* Counterfactual Cosine Similarity Score `[Bouchard, 2024] <https://arxiv.org/abs/2407.10853>`_
* Counterfactual BLEU `[Bouchard, 2024] <https://arxiv.org/abs/2407.10853>`_
* Counterfactual ROUGE-L `[Bouchard, 2024] <https://arxiv.org/abs/2407.10853>`_

**Stereotype Metrics** 

* Stereotypical Associations `[Liang et al., 2023] <https://arxiv.org/abs/2211.09110>`_
* Co-occurrence Bias Score `[Bordia & Bowman, 2019] <https://arxiv.org/abs/1904.03035>`_
* Stereotype classifier metrics `[Zekun et al., 2023] <https://arxiv.org/abs/2311.14126>`_, `[Bouchard, 2024] <https://arxiv.org/abs/2407.10853>`_

**Recommendation (Counterfactual) Fairness Metrics**

* Jaccard Similarity `[Zhang et al., 2023] <https://dl.acm.org/doi/10.1145/3604915.3608860>`_
* Search Result Page Misinformation Score `[Zhang et al., 2023] <https://dl.acm.org/doi/10.1145/3604915.3608860>`_
* Pairwise Ranking Accuracy Gap `[Zhang et al., 2023] <https://dl.acm.org/doi/10.1145/3604915.3608860>`_

**Classification Fairness Metrics**

* Predicted Prevalence Rate Disparity `[Feldman et al., 2015] <https://arxiv.org/abs/1412.3756>`_, `[Bellamy et al., 2018] <https://arxiv.org/abs/1810.01943>`_, `[Saleiro et al., 2019] <https://arxiv.org/abs/1811.05577>`_
* False Negative Rate Disparity `[Bellamy et al., 2018] <https://arxiv.org/abs/1810.01943>`_, `[Saleiro et al., 2019] <https://arxiv.org/abs/1811.05577>`_
* False Omission Rate Disparity `[Bellamy et al., 2018] <https://arxiv.org/abs/1810.01943>`_, `[Saleiro et al., 2019] <https://arxiv.org/abs/1811.05577>`_
* False Positive Rate Disparity `[Bellamy et al., 2018] <https://arxiv.org/abs/1810.01943>`_, `[Saleiro et al., 2019] <https://arxiv.org/abs/1811.05577>`_
* False Discovery Rate Disparity `[Bellamy et al., 2018] <https://arxiv.org/abs/1810.01943>`_, `[Saleiro et al., 2019] <https://arxiv.org/abs/1811.05577>`_

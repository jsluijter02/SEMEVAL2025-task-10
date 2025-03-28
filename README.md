# SemEval 2025 Task 10 (Task 2): Multilingual Characterization and Extraction of Narratives from Online News
This repository contains the code for an NLP project aiming to classify dominant narratives from multilingual news articles using techniques such as logistic regression, SVMs, and the OpenAI API.
It is a solution to Task 2: Narrative Classification of the SemEval 2025 Task 10 shared task.
Task 2 involves assigning each news article a dominant narrative label from a predefined taxonomy. The task is multilingual, covering English, Portuguese, Hindi, Russian, and Bulgarian, and focuses on two domains: the Ukraine-Russia Conflict and Climate Change. 

## Task Desctription
### Task 2: Narrative Classification
Given a news article and a predefined taxonomy of narratives and subnarratives, the goal is to classify the article under its most relevant narrative label.
The task spans five languages:
- English
- Portuguese
- Bulgarian
- Hindi
- Russian
### Narrative Labels
Each article is annotated with one or more subnarrative labels, which are specific narrative themes tied to broader dominant narratives. In this project, the goal is to classify all relevant subnarratives for a given article.
Since each subnarrative is directly linked to a dominant narrative, the dominant category can be inferred automatically once subnarratives are predicted.
This allows for more straight-forward classification while preserving hierarchical narrative structure.

## Repository Structure
<pre>SEMEVAL2025-task-10/
├── data/                       - Contains the raw narrative/news article data files and the preprocessed train/test data objects.
├── erroranalysis/              - Contains the results of the error analysis script.
├── notebooks/                  - Contains the experiment notebooks.
│   ├── LLM_notebook.ipynb           - Experiments with the OpenAI API to classify the narratives.
│   ├── SBertEmbeddings.ipynb        - Generates Sentence-BERT embeddings as a richer alternative to TF-IDF.
│   ├── SVC_notebook.ipynb           - Experiments using an SVM Classifier (SVC) for narrative classification.
│   ├── data_analysis.ipynb          - Exploratory data analysis prior to classification.
│   └── data_process-and-save.ipynb - Loads and combines text files and labels into a single DataFrame.
├── pkl_files/                 - Contains the `MultiLabelBinarizer` and SBERT embedding objects.
├── src/                       - Source code for the pipeline.
│   ├── eval.py                     - Evaluation and error analysis classes.
│   ├── model.py                    - Model class definitions.
│   ├── pipeline.py                 - Pipeline class that runs classification and analysis.
│   ├── postprocess.py              - Writes model predictions to SemEval-formatted output.
│   ├── preprocess.py               - Data preprocessing (TF-IDF or embeddings) and train/test split.
│   └── utils.py                    - Shared loading utilities used throughout the project.<pre>

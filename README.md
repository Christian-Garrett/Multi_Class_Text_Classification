### AG News Text Classification with BERT ###

This project implements text classification on the AG News dataset using BERT transformer model.
AG News (AG’s News Corpus) is a sub dataset of AG's corpus of news articles constructed by assembling titles and description fields of articles, from the 4 largest classes (“World”, “Sports”, “Business”, “Sci/Tech”) of AG’s Corpus.

BERT is a state-of-the-art transformer model use to perform complex NLP tasks like NLU, NLG, Sentiment Analysis, Text Classification, etc. that gives very high performance metrics, sometimes even surpassing human capabilities.

Below are the steps to be followed:

1. Install the required packages stated in requirements.txt file 
   Packages can be installed on an Anaconda environment or on normal python interpreter.
   For Anaconda:
   conda create --name <youenvname>
   conda activate <yourenvname>
   pip install -r requirements.txt
   For Python Interpreter:
   pip install -r requirements.txt
   
2. There are 2 primary packages:
   a> ML_Pipeline:
   Project modules to perform specific Machine Learning task.
   b> engine.py:
   Project function calls are done here.
   
3. Run/Debug the engine.py file to execute the code logic.

4. All input datasets are stored in the input folder.

5. All predictions and models are stored in the output folder (gitignored due to size).



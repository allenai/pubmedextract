# PubMed-Extract

Quantifying demographic bias in clinical trials using corpus of academic papers. 

This repo accompanies the paper: Quantifying Sex Bias in Clinical Studies at Scale with Automated Data Extraction by
Sergey Feldman, Waleed Ammar, Kyle Lo, Elly Trepman, Madeleine van Zuylen, and Oren Etzioni.

This code takes as input a clinical trial paper parsed by Omnipage and returns the extracted number
of participating women and men.

This package is being released for (a) algorithmic documentation and (b) statistical analysis reproduction purposes,
and will not work for extracting clinical trial participant counts from new PDFs you may have as it depends on
Omnipage. For an example of the type of input that PubMed-Extract expects, see `tests/test_sex/papers/`.

The code for (a) is located in `pubmedextract/` The code for (b) is located in `analysis_scripts/`.

Note that the paper compares PubMed-Extract to an algorithm referred to as AACT-Query.
This is a relatively simple algorithm and its execution (essentially a SQL query) is contained entirely within `analysis_scripts/04_analysis.py`.


## Installation

This project requires **Python 3.6**.  We recommend you set up a conda environment:
 
```
conda create -n pubmedextract python=3.6
conda activate pubmedextract
conda install spacy=1.9 thinc=6.5.2 matplotlib=3.0.1 seaborn=0.9.0 joblib=0.12.0 psycopg2=2.7.5 pandas=0.23.4 statsmodels=0.9.0 patsy pytest pylint
```
You may need to do `source activate pubmedextract` instead of `conda activate pubmedxtract`, depending on your `anaconda` version.


Then clone the repo, and install it (along with remaining requirements).

```
git clone https://github.com/allenai/pubmedextract.git
cd pubmedextract
python setup.py install
```


## Tests

After installing, you can run all the unit tests:

```
pylint --disable=R,C,W pubmedextract
python -m pytest tests/
```


## Usage Example: Extracting Gender Counts from Available JSON Inputs
A simple example is in `scripts/parse_paper_example.py`, and also reproduced in its entirety below:

```
import pickle
from pubmedextract.sex import get_sex_counts
from pubmedextract.table_utils import PaperTable

# load some example papers
# assumes the cwd is pubmedextract/
with open('tests/test_sex/test_papers_and_counts.pickle', 'rb') as f:
    s2ids_and_true_counts, _ = pickle.load(f)

# get the counts and print them out
for s2id, true_counts in s2ids_and_true_counts:
    paper = PaperTable(s2id, 'tests/test_sex/papers/')
    demographic_info = get_sex_counts(paper)
    print('True counts:', true_counts)
    print('Estimated counts:', demographic_info.counts_dict, '\n')
```


## Paper Analysis Reproduction
The scripts needs to reproduce the analyses in the paper `Quantifying Sex Bias in Clinical Studies at Scale with Automated Data Extraction` can be found here: https://github.com/allenai/pubmedextract/tree/master/analysis_scripts.

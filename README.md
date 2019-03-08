# PubMedExtract

Quantifying demographic bias in clinical trials using corpus of academic papers.

This code takes as input a clinical trial paper parsed by Omnipage and returns the extracted number
of participating women and men.

This package is being released for (a) algorithmic documentation and (b) statistical analysis reproduction purposes,
and will not work for extracting clinical trial participant counts from new PDFs you may have as it depends on
Omnipage. For an example of the type of input that PubMed-Extract expectsh, see `tests\test_sex\papers\`


## Extracting Gender Counts from Available JSON Inputs
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


## Installation

This project requires **Python 3.6**.  We recommend you set up a conda environment:
 
```
conda create -n pubmedextract python=3.6
source activate pubmedextract
```

The dependencies are listed in the `requirements.in` file:

```
pip install -r requirements.in
```


## Tests

After installing, you can run all the unit tests:

```
pylint --disable=R,C,W pubmedextract
pytest tests/
```

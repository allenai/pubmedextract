# PubMedExtract

TODO: update this file after making a public repo.

Quantifying demographic bias in clinical trials using corpus of academic papers.

This code takes as input a clinical trial paper parsed by Omnipage and returns the extracted number
of participating women and men. The algorithm will often return `nan` counts, 
indicating that it failed to get a confident estimate.

## Generating the Input
The inputs to the algorithm are generated as follows.

First, write the list of S2 paper IDs in a text file `ids.txt` (one paper ID per line) to fetch them from S3:
```
cd scholar-research/corvid/
python scripts/bulk_fetch_pdfs_from_s3.py -p ids.txt
```

The PDFs will be downloaded to a local location that can be viewed in the `configs.py` file at the root of `scholar-research/corvid/`, currently set as follows:
```
import os
from corvid.util.files import canonicalize_path

DATA_DIR = canonicalize_path('/net/nfs.corp/s2-research/corvid/')
PAPERS_DIR = os.path.join(DATA_DIR, 'papers/')
```

According to this configuration, the PDFs would be saved to `/net/nfs.corp/s2-research/corvid/papers/<paper_id>/<paper_id>.pdf`

Subsequently, extract the tables from each PDF as follows:
```
python scripts/bulk_extract_tables_from_pdfs_via_omnipage.py -p ids.txt
```
This will look for the PDF in the location specified in the config, pass it to Omnipage to get XML, extract tables and serialize them to JSON.  The XML and JSON are saved in the same location as the PDF.


## Extracting Gender Counts from the Input
A simple example is in the `scripts` folder, and also reproduced in its entirety below:

```
import pickle
from pubmedextract.sex import get_sex_counts
from pubmedextract.table_utils import PaperTable

# load some example papers
# assumes the cwd is scholar_research/experiments/pubmedextract
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

This project requires **Python 3.6** or greater.  

To install this project, first follow the installation instructions in `https://github.com/allenai/scholar-research/tree/master/corvid`,
and then the following (when `pubmedextract` is the current working directory):

```
source activate s2-corvid
pip install -r requirements.in
python setup.py install
```


## Tests

After installing, you can run all the unit tests:

```
pylint --disable=R,C,W pubmedextract
pytest tests/
```

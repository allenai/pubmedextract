# Code and Data for 'Quantifying Sex Bias in Clinical Studies at Scale with Automated Data Extraction'
This collection of scripts and data enables the reproduction of results, tables, and figures from the paper
'Quantifying Sex Bias in Clinical Studies at Scale with Automated Data Extraction'. 

## Scripts
The script files are as follows:

* `01_subet_pubmed_to_clinical_trials.py` - This file provides code that was used to subset the entire 
PubMed archive to records that are related to clinical trials. Note that the records were then further
subsetted to those for which Semantic Scholar had full PDFs, but the code for this process is not included.
To run this file, you will need to download the entire PubMed XML corpus (ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline).
See comments within the script for more information.
* `02_data_setup.py` - This file contains code to pre-process MeSH data, and obtain mappings from PubMed
records to disease-related MeSH terms.
* `03_apply_pubmedextract.py` - Code for the application of PubMedExtract algorithm. This takes a very long time. 
If you wish to skip this, the results are already included in `data/pubmedextract_results.pickle`.
* `04_analysis.py` - Code to generate statistical analysis, tables, and figures.

Once the parent package `pubmedextract` is installed via `python setup.py pubmedextract` and the data 
needed is downloaded (see below), scripts 02, 03, and 04 should be executed in order while the current working directory is `pubmedextract/scripts`

## Data

The files required for the above scripts are available in the following public S3 bucket: `s3://ai2-pubmedextract`.

The 3.2GB of files should be downloaded into this folder: `pubmedxtract/scripts/data` via the AWS CLI: 

`aws s3 sync s3://ai2-pubmedextract/ pubmedxtract/scripts/data`

If you intend to rerun `pubmedxtract/scripts/03_apply_pubmedextract.py`, then unarchive the file `papers_json.gz.tar` into the same folder. 
It should create a new folder: `pubmedextract/scripts/data/papers_json`. Both the unarchiving and script take many hours.

The files/folders are as follows:

* `aact_query_jan_7_2019.pickle` - The results of the commented out AACT query in `04_analysis.py`. This was run at the end of 2018 and is included for reproducibility.  
* `papers_json.gz.tar` - An archive containing the folder with a large number of paper table files (in json format) that are required for reproduction. 
The contents are necessary for `03_apply_pubmedextract.py`.
* `disease_category_prevalence_global.tsv` - 2016 global disease prevalence data obtained via the [GHDx](http://ghdx.healthdata.org/gbd-results-tool).
* `mesh2category_global.tsv` - MeSH to disease category mapping.
* `mtrees2018.bin` - MeSH tree hierarchy available [here](ftp://nlmpubs.nlm.nih.gov/online/mesh/MESH_FILES/meshtrees/).
* `pubmed_id_mesh_map.pickle` - A map from PubMed IDs to MeSH terms. Generated by `02_data_setup.py`.
* `pubmed_s2_data.pickle` - PubMed records that also have PDFs in Semantic Scholar. 
* `pubmed_s2_id_maps.pickle` - A map from PubMed IDs to Semantic Scholar IDs. 
* `pubmedextract_results.pickle` - Results from applying PubMedExtract algorithm to papers in Semantic Scholar. Generated by `03_apply_pubmedextract.py`.
* `sample.xml` - A sample MEDLINE XML file to be used with `01_subet_pubmed_to_clinical_trials.py`.
* `trial_n_vs_bias.png` - A paper figure generated by `04_analysis.py`.
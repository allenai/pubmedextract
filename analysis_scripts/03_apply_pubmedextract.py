import pickle
import numpy as np
from pubmedextract.sex import get_sex_counts
from pubmedextract.table_utils import PaperTable
from joblib import Parallel, delayed

PAPERS_DIR = 'data/papers_json'
N_JOBS = 8  # set to number of processes available

# load pubmed paper data that is also in S2
with open('data/pubmed_s2_data.pickle', 'rb') as f:
    pubmed_s2_data = pickle.load(f)

with open('data/pubmed_s2_id_maps.pickle', 'rb') as f:
    s2id_to_pmid, pmid_to_s2id = pickle.load(f)

with open('data/pubmed_id_mesh_map.pickle', 'rb') as f:
    pmid_to_mesh = pickle.load(f)

# get a mapping from mesh term to disease category
mesh_to_category = {}
with open('data/mesh2category_global.tsv', 'r') as f:
    for line in f:
        mesh, cat = line.strip().split('\t')
        if cat != 'None of the above':
            mesh_to_category[mesh] = cat.strip('*')


def get_disease_categories(pmid):
    categories = []
    if pmid in pmid_to_mesh:
        categories = list(set([
            mesh_to_category[i]
            for i in pmid_to_mesh[pmid]
            if i in mesh_to_category
        ]))
    return categories


def apply_mednir(pmid):
    # possible failure modes:
    # (1) we don't have a mapping from pmid to s2id
    # (2) we have a mapping, but we have no Paper
    # (3) we have a paper but we have no tables
    # (4) we have tables, but mednir failed
    if pmid in pmid_to_s2id:
        s2id = pmid_to_s2id[pmid]
        paper = PaperTable(s2id, PAPERS_DIR)
        if paper.tables is None:
            return 'no_paper'
        elif len(paper.tables) == 0:
            return 'no_tables'
        else:
            demographic_info = get_sex_counts(paper)
            return demographic_info.counts_dict
    else:
        return 'no_s2id'


# keep only pmids that have disease category mesh terms
pubmed_s2_data_keys = []
for pmid in pubmed_s2_data.keys():
    if pmid in pmid_to_mesh:
        categories = list(set([
            mesh_to_category[i]
            for i in pmid_to_mesh[pmid]
            if i in mesh_to_category
        ]))
        if len(categories) > 0:
            pubmed_s2_data_keys.append(pmid)

# run in parallel
joblib_results = Parallel(n_jobs=N_JOBS, verbose=5, backend='threading')(
    delayed(apply_mednir)(pmid) for pmid in pubmed_s2_data_keys
)

# combine results from parallel runs and exclude the various errors
results_all = {key: val for key, val in zip(pubmed_s2_data_keys, joblib_results) if type(val) is not str}

# keep only results that have non-missing numerical results and also add in year
results_valid = {}
for key, val in results_all.items():
    if ~np.isnan(val['males']):
        val['year'] = pubmed_s2_data[key]['year']
        results_valid[key] = val

# save
with open('data/pubmedextract_results.pickle', 'wb') as f:
    pickle.dump(results_valid, f)

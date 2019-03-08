import pickle
import numpy as np
from collections import Counter


# load pubmed paper data that is also in S2
with open('data/pubmed_s2_data.pickle', 'rb') as f:
    pubmed_s2_data = pickle.load(f)

with open('data/pubmed_s2_id_maps.pickle', 'rb') as f:
    s2id_to_pmid, pmid_to_s2id = pickle.load(f)

# import mesh
# note there are multiple tree paths for each one
term2mesh = {}
mesh2term = {}
with open('data/mtrees2018.bin', 'r') as f:
    for line in f:
        term, meshid = line.strip().split(';')
        meshid = tuple(meshid.split('.'))
        if term not in term2mesh:
            term2mesh[term] = []
        term2mesh[term].append(meshid)
        mesh2term[meshid] = term

# for each term, figure out of its parent has relevant key words:
# disease, vaccination, disorder, pathological or neoplasms
disease_match = {}
for term, meshids in term2mesh.items():
    meshids_cum = [[meshid[:i] for i in range(1, len(meshid) + 1)]
                   for meshid in meshids]
    tree_paths = [[mesh2term[j].lower() for j in js]
                  for js in meshids_cum]
    has_terms = [['disease' in i or
                  'vaccination' in i or
                  'disorder' in i or
                  'pathological' in i or
                  'neoplasms' in i
                  for i in tree_path]
                 for tree_path in tree_paths]
    disease_match[term] = np.any(np.any(has_terms))

# get all the most common important mesh terms in pubmed_s2_data
counter = Counter()
pmid_to_mesh = {}
has_disease_count = 0
for pmid, trial_result in pubmed_s2_data.items():
    if trial_result['meshlist'] is not None:
        # find disease mesh terms first
        disease_mesh = np.array(
            [i for i in trial_result['meshlist']
             if i[0] in disease_match
             and disease_match[i[0]]]
        )
        if len(disease_mesh) > 0:
            # if there are important terms, just get those
            disease_mesh_y = disease_mesh[disease_mesh[:, 2] == 'Y', 0]
            if len(disease_mesh_y) > 0:
                disease_mesh = disease_mesh_y
            else: # otherwise, take all disease terms
                disease_mesh = disease_mesh[:, 0]
            counter.update(disease_mesh)
            has_disease_count += 1
            pmid_to_mesh[pmid] = disease_mesh

with open('data/pubmed_id_mesh_map.pickle', 'wb') as f:
    pickle.dump(pmid_to_mesh, f)

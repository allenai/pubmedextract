from joblib import Parallel, delayed
from xml.etree import ElementTree as etree
import pandas as pd

N_JOBS = 8 # number of processes available on your machine

# replace with a list of MEDLINE/PubMed XML files
# which are available here: ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline
# and here: ftp://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/
# note that to run this script for the full XML archive will require
# a huge amount of ram, and this code is included largely
# for documentation purposes
files = ['data/sample.xml']

valid_pubtypes = {
    'Adaptive Clinical Trial',
    'Clinical Study',
    'Clinical Trial',
    'Clinical Trial, Phase I',
    'Clinical Trial, Phase II',
    'Clinical Trial, Phase III',
    'Clinical Trial, Phase IV',
    'Collected Works',
    'Controlled Clinical Trial',
    'Equivalence Trial',
    'Multicenter Study',
    'Observational Study',
    'Pragmatic Clinical Trial',
    'Randomized Controlled Trial',
    'Study Characteristics',
    'Twin Study',
    'Validation Studies'
}

'''
Step 1: Get all of the papers from MEDLINE database, their ids, their year, and their publication type
'''
def process_file(file_name):
    try:
        return (get_links_from_file(file_name), file_name)
    except Exception as e:
        return (e, file_name)


def get_links_from_file(file_name):
    """ Go through a XML file and find the pubmed, pmc, and various database ids
        And also find all the article types
    """
    root = etree.fromstringlist(list(open(file_name, encoding='utf8')))
    paperinfos = []
    for child in root:
        if child[0].text is None:
            databaseids = set()
            pubtypelist = []
            meshlist = []
            pmid = child[0][0].text
            pmcid = None
            year = None
            for elem in child.iterfind('PubmedData/ArticleIdList/ArticleId'):
                if elem.text is not None and elem.text.lower().startswith('pmc'):
                    pmcid = elem.text # should only be one of these
            for elem in child.iterfind('MedlineCitation/Article/DataBankList/DataBank/AccessionNumberList/AccessionNumber'):
                if elem.text is not None:
                    databaseids.add(elem.text)
            for elem in child.iterfind('MedlineCitation/Article/PublicationTypeList/'):
                if elem.text is not None:
                    pubtypelist.append(elem.text)
            for elem in child.iterfind('MedlineCitation/MeshHeadingList/MeshHeading/DescriptorName'):
                if elem.text is not None:
                    d = elem.attrib
                    d['mesh_name'] = elem.text
                    meshlist.append(d)
            for elem in child.iterfind('MedlineCitation/Article/Journal/JournalIssue/PubDate/Year'):
                if elem.text is not None:
                    year = elem.text
            meshlist = pd.DataFrame.from_records(meshlist)
            if 'mesh_name' in meshlist.columns:
                meshlist = meshlist[['mesh_name', 'UI', 'MajorTopicYN']].values.astype(str)
            else:
                meshlist = None
            paperinfo = {'pmid': pmid, 'databaseids': databaseids, 'pmcid': pmcid, 'pubtypelist': pubtypelist,
                         'meshlist': meshlist, 'year': year}
            paperinfos.append(paperinfo)
    return paperinfos

list_of_results = Parallel(n_jobs=N_JOBS, verbose=25)(delayed(process_file)(file) for file in files)

# some may have errored out - we catch those here and redo
not_done_files = [i[1] for i in list_of_results if type(i[0]) is not list]
for file_name in not_done_files:
    print('Processing:', file_name)
    result = process_file(file_name)
    if type(result[0]) is list:
        list_of_results.append(result)
        print('Success!')
    else:
        print('Failed!')

# combine results from list_of_results
results = []
for result, file_name in list_of_results:
    if type(result) is list:
        results.extend(result)

# we now filter down to only those that are trials of some kind
trial_results = [i for i in results if len(set(i['pubtypelist']).intersection(valid_pubtypes)) > 0]

'''
Note: at this point we use internal S2 processes to subset trial_results to only include papers
for which Semantic Scholar has a PDF available. This results in the file data/pubmed_s2_data.pickle
'''
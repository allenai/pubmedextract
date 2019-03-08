import pickle
import unittest
from pubmedextract.table_utils import PaperTable
from pubmedextract.sex import get_sex_counts


class TestSexExtractionCorrectness(unittest.TestCase):
    def setUp(self):
        # these are the papers from Madeleine's annotations:
        # the 90% subset which are extracted correctly
        with open('tests/test_sex/test_papers_and_counts.pickle', 'rb') as f:
            self.s2ids_and_true_counts, self.s2ids_wrong = pickle.load(f)
        
    def test(self):
        local_papers_dir = 'tests/test_sex/papers/'
        for s2id, true_counts in self.s2ids_and_true_counts:
            paper = PaperTable(s2id, local_papers_dir)
            demographic_info = get_sex_counts(paper)
            if paper.id not in self.s2ids_wrong:
                assert demographic_info.counts_dict == true_counts

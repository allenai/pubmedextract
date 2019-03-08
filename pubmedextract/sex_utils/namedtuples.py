from collections import namedtuple

Id = namedtuple('Id', ['pmid', 'aactid', 'pmcid', 's2id'])

DemographicInfo = namedtuple('DemographicInfo', ['counts_dict',
                                                 'counts_notification',
                                                 'parsed_tables',
                                                 'parsed_tables_notifications'])

NumericalCell = namedtuple('NumericalCell', ['n1', 'n2', 'parens'])

ParsedCell = namedtuple('ParsedCell',
                        ['row_header', 'col_header', 'cell_val',
                         'parsed_cell_val', 'n_equals_row',
                         'n_equals_col', 'row', 'col', 'male_counts',
                         'female_counts', 'extract_msg', 'cell_type_from_row'])

ParsedTable = namedtuple('ParsedTable', ['row_headers', 'col_headers', 'counts_matrix_headers', 'counts_matrix'])

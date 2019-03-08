import numpy as np

from pubmedextract.sex_utils.namedtuples import DemographicInfo
from pubmedextract.sex_utils.parse_sex_rows import parse_sex_rows
from pubmedextract.sex_utils.subdivide_table import subdivide
from pubmedextract.sex_utils.sex_from_omnipage_tables import extract_male_female_counts_from_tables


def get_sex_counts(paper):
    # format each table in a paper
    if paper.tables is None or len(paper.tables) == 0:
        parsed_tables = []
        parsed_tables_notifications = []
        counts_notification = 'omnipage did not extract any tables'
    else:
        parsed_tables = []
        parsed_tables_notifications = []
        for table in paper.tables:
            if table.ncol == 1 or table.nrow == 1:
                table_notification = 'table has 1 column or row'
            else:
                column_headers, row_headers, numerical_subtable = subdivide(table)
                if len(numerical_subtable) == 0:
                    table_notification = 'table has no numerical subtable'
                elif len(row_headers) == 0:
                    table_notification = 'table has no row headers'
                else:
                    parsed_table = parse_sex_rows(column_headers, row_headers, numerical_subtable)
                    if parsed_table is not None and len(parsed_table) > 0:
                        table_notification = 'table sex cells extracted'
                        parsed_tables.append(parsed_table)
                    else:
                        table_notification = 'no sex cells in table'
            parsed_tables_notifications.append(table_notification)
        if len(parsed_tables) == 0:
            counts_notification = 'none of the tables have sex info OR the tables were badly parsed by omnipage'

    # extract counts from each formatted table
    if len(parsed_tables) > 0:
        counts, counts_notification = extract_male_female_counts_from_tables(parsed_tables)
        if counts is not None:
            counts_dict = {'males': int(counts[0]), 'females': int(counts[1])}
        else:
            counts_dict = {'males': np.nan, 'females': np.nan}
    else:
        counts_dict = {'males': np.nan, 'females': np.nan}

    demographic_info = DemographicInfo(counts_dict, counts_notification, parsed_tables, parsed_tables_notifications)

    return demographic_info

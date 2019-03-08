import numpy as np
from pubmedextract.sex_utils.constants import SEX_VOCAB
from pubmedextract.sex_utils.namedtuples import ParsedTable, ParsedCell
from pubmedextract.sex_utils.regex_utils import extract_n_equals_value, regex_numerical_field

import spacy

tokenizer = spacy.load('en', disable=['parser', 'ner', 'tagger'])

def parse_sex_rows(column_headers, row_headers, numerical_subtable):
    """
    Extract rows that have sex cells in them
    Format sex rows into a 3d np.array
    """
    # extract sex rows
    row_headers_tokenized = [[str(i).lower() for i in tokenizer(s)] for s in row_headers]
    col_headers_tokenized = [[str(i).lower() for i in tokenizer(s)] for s in column_headers]
    nrows, ncols = numerical_subtable.shape
    sex_cells = []
    for row in range(nrows):
        if len(SEX_VOCAB.intersection(row_headers_tokenized[row])) > 0:
            result_row = []
            for col in range(ncols):
                cell_val = numerical_subtable[row, col]
                row_header = row_headers_tokenized[row]
                col_header = col_headers_tokenized[col]
                n_equals_row = extract_n_equals_value(row_headers[row])
                n_equals_col = extract_n_equals_value(column_headers[col])
                parsed_cell_val = regex_numerical_field(cell_val, row_header, col_header, n_equals_row, n_equals_col)
                participant_counts = extract_sex_count_from_parsed_cell(row_header,
                                                                        col_header,
                                                                        parsed_cell_val,
                                                                        n_equals_col,
                                                                        n_equals_row)

                parsed_cell = ParsedCell(row_header=row_header,
                                         col_header=col_header,
                                         cell_val=cell_val,
                                         parsed_cell_val=parsed_cell_val,
                                         n_equals_row=n_equals_row,
                                         n_equals_col=n_equals_col,
                                         row=row,
                                         col=col,
                                         male_counts=participant_counts['male'],
                                         female_counts=participant_counts['female'],
                                         cell_type_from_row=participant_counts['cell_type_from_row'],
                                         extract_msg=participant_counts['error'])
                result_row.append(parsed_cell)
            error_frac = np.mean([i.extract_msg.startswith('error') for i in result_row])
            if len(result_row) > 0 and error_frac < 1:
                sex_cells.append(result_row)

    if len(sex_cells) == 0:
        return None

    # format results thus far into 3d np.array named counts_matrix
    # it will be sized 4 x n_rows x n_cols
    # where the 4 will stand for counts: male, female, males estimated, female estimated
    counts_matrix_headers = ['M', 'F', 'Mest', 'Fest']
    row_headers = np.array([i[0].row_header for i in sex_cells])
    col_headers = np.array([i.col_header for i in sex_cells[0]])
    counts_matrix = np.zeros((4, len(row_headers), len(col_headers)), dtype=np.int) * np.nan
    for row in range(len(row_headers)):
        for col in range(len(col_headers)):
            cell = sex_cells[row][col]
            if cell.cell_type_from_row == 'mf' or cell.cell_type_from_row == 'fm':
                counts_matrix[counts_matrix_headers.index('M'), row, col] = cell.male_counts
                counts_matrix[counts_matrix_headers.index('F'), row, col] = cell.female_counts
            elif cell.cell_type_from_row == 'male':
                counts_matrix[counts_matrix_headers.index('M'), row, col] = cell.male_counts
                counts_matrix[counts_matrix_headers.index('Fest'), row, col] = cell.female_counts
            elif cell.cell_type_from_row == 'female':
                counts_matrix[counts_matrix_headers.index('Mest'), row, col] = cell.male_counts
                counts_matrix[counts_matrix_headers.index('F'), row, col] = cell.female_counts

    # get rid of columns that are all nans for M and F
    nan_males = np.isnan(counts_matrix[0, :, :])
    nan_females = np.isnan(counts_matrix[1, :, :])
    nan_cols = np.bitwise_and(nan_males, nan_females).mean(0) == 1
    col_headers = col_headers[~nan_cols]
    counts_matrix = counts_matrix[:, :, ~nan_cols]

    parsed_table = ParsedTable(row_headers=row_headers,
                               col_headers=col_headers,
                               counts_matrix_headers=counts_matrix_headers,
                               counts_matrix=counts_matrix)

    return parsed_table


def extract_sex_count_from_parsed_cell(row_header, col_header, parsed_cell_val, n_equals_col, n_equals_row):
    """
    The big function to process a single cell to extract male/female counts.

    If a row contains only male counts, the code tries to estimate female counts using other available info such as
    n_equals_col, n_equals_row, and percent values if available. This is useful downstream when trying to extract a single
    consistent estimate of sex count from tables with variable number of rows and multiple tables.

    Returns a dictionary with these keys:
        'male': count extracted for male
        'female': count extracted for female
        'error': some descriptive error or notification or warning
        'cell_type_from_row':
            'm' - row contains male count (female count is estimated if available)
            'f' - row contains female count (male count is estimated if available)
            'mf' - row contains both male and female counts (male is first value, female is second)
            'fm' - row contains both male and female counts (female is first value, male is second)
    """
    # this prevents all subsequent code from working so catch early
    if parsed_cell_val is None:
        return {'male': np.nan, 'female': np.nan, 'error': 'error: no parsed_cell_value', 'cell_type_from_row': None}

    # for convenience
    n_equals = n_equals_col or n_equals_row
    n1 = parsed_cell_val.n1
    n2 = parsed_cell_val.n2
    parens = parsed_cell_val.parens

    # none of these should be numbers less than 1. percents are always out of 100
    if n_equals is not None and n_equals < 1:
        return {'male': np.nan, 'female': np.nan, 'error': 'error: n_equals is less than 1', 'cell_type_from_row': None}
    elif n1 is not None and 0 < n1 < 1:
        return {'male': np.nan, 'female': np.nan, 'error': 'error: n1 is less than 1', 'cell_type_from_row': None}
    elif n2 is not None and 0 < n2 < 1:
        return {'male': np.nan, 'female': np.nan, 'error': 'error: n2 is less than 1', 'cell_type_from_row': None}
    elif parens is not None and (parens < 1 or parens > 100):
        return {'male': np.nan, 'female': np.nan, 'error': 'error: parens is less than 1 or greater than 100',
                'cell_type_from_row': None}

    # catch common rows in col words that are very often wrong
    suspicious_col_words = ['ratio', 'p', 'pvalue', 'p-value', 'chi', 'odds',
                            '-value', 'event', 'rate', 'fraction', 'concentration',
                            'cost', 'costs', 'years', 'year']
    suspicious_word_in_col_header = np.any([i in col_header for i in suspicious_col_words])

    # only a certain set of words are allowed to be in the row header
    # if more than that appear, it's probably not a count
    allowed_row_words = {
        ' ', '%', '(', ')', '*', ',', '-', '—', '–', '-n', '.', '/', ':', ';', '=', '[', ']', 'and', 'characteristics',
        'demographic', 'demographics', 'enrollment', 'f', 'female', 'females', 'gender', 'm', 'male', 'males',
        'men', 'n', 'n(%', 'no', 'no./total', 'number', 'of', 'patient', 'patients', 'percent', 'proportion',
        'sex', 'subjects', 'to', 'total', 'women'
    }
    row_header_set = set(row_header)
    only_allowed_words = len(row_header_set.intersection(allowed_row_words)) == len(row_header_set)

    if suspicious_word_in_col_header or not only_allowed_words:
        return {'male': np.nan, 'female': np.nan,
                'error': 'error: row or col header suggests not participant count',
                'cell_type_from_row': None}

    #  n_equals shouldn't be smaller than n1 or n2 (if they are int)
    if n_equals is not None:
        if type(n1) is int and n_equals < n1:
            n_equals = None
        elif type(n2) is int and n_equals < n2:
            n_equals = None

    # figure out of row is about males or females or both
    row_male_index = [row_header.index(i) for i in ['male', 'males', 'men'] if i in row_header]
    row_male_index = row_male_index[0] if row_male_index else None
    row_female_index = [row_header.index(i) for i in ['female', 'females', 'women'] if i in row_header]
    row_female_index = row_female_index[0] if row_female_index else None
    row_has_both = (row_male_index is not None) and (row_female_index is not None) and (n1 is not None) and (
                     n2 is not None)

    # catch more bad cases early and return
    if type(n1) is not float and type(n1) is not int:
        return {'male': np.nan, 'female': np.nan, 'error': 'error: n1 is missing', 'cell_type_from_row': None}
    elif type(n1) is float and type(parens) is float:
        return {'male': np.nan, 'female': np.nan, 'error': 'error: n1 and parens are both float', 'cell_type_from_row': None}
    elif type(n1) is float and n_equals is None:
        return {'male': np.nan, 'female': np.nan, 'error': 'error: n1 is float, but there is no n_equals',
                'cell_type_from_row': None}

    # sometimes with n1 / n2 for rows that don't have both we have that n2 != n_equals, but they should be at least close
    if not row_has_both and type(n1) is int and type(n2) is int and n_equals is not None and (n_equals != n2):
        if n2 < n_equals and (n_equals - n2) / n_equals <= 0.1: # within ten percent
            n_equals = None
        else:
            return {'male': np.nan, 'female': np.nan,
                    'error': 'error: n1, n2, n_equals are all available for single sex row but n2 != n_equals',
                    'cell_type_from_row': None}

    # make parens a fraction for later calculations
    parens_type = type(parens)
    if parens is not None:
        parens /= 100

    # case where both male and female are present
    if row_has_both:
        if type(n1) is int and type(n2) is int:
            first_int = n1
            second_int = n2
            error = 'notification: n1, n2 are int'
        elif n_equals is not None and type(n1) is float and type(n2) is float:
            first_int = int(np.round(n1 / 100 * n_equals))
            second_int = int(np.round(n2 / 100 * n_equals))
            error = 'notification: n1, n2 are float and are assumed to be percentages, n_equals available'
        else:
            first_int = np.nan
            second_int = np.nan
            error = 'error: row has both male & female, but n1/n2 are either not available or have mismatched types'

        if row_male_index < row_female_index:
            return {'male': first_int, 'female': second_int, 'error': error, 'cell_type_from_row': 'mf'}
        else:
            return {'male': second_int, 'female': first_int, 'error': error, 'cell_type_from_row': 'fm'}
    # case where only one of the two is present
    elif row_male_index is not None or row_female_index is not None:
        if n_equals:
            if type(n1) is float:  # this means n1 is actually a parens
                first_int = int(np.round(n1 / 100 * n_equals))
                if n2 is not None:  # not sure why this would ever happen for a row with only male OR female
                    second_int = np.nan
                    error = 'error: single male OR female row, n1 is float AND n2 is available'
                else:
                    second_int = int(np.round(n_equals * (1.0 - n1 / 100)))
                    error = 'notification: n_equals available, n1 is float, n2 is estimated'
            elif type(n1) is int:
                first_int = n1
                second_int_est = n_equals - n1
                if parens:
                    if parens_type is float:
                        # check if n_equals is correct
                        n_equals_est = int(np.round(n1 / parens))
                        dist_absolute = np.abs(n_equals - n_equals_est)
                        dist_relative = 100 * dist_absolute / n_equals
                        if dist_relative <= 5 or dist_absolute <= 5:
                            second_int = second_int_est
                            error = 'notification: n2 estimates agreed to within tolerance (5 participants and 5%)'
                        else:
                            second_int = int(np.round(n1 / parens - n1)) # maybe return nan if turns out to be source of error
                            error = 'warning: n2 estimates did not agree'
                    else:
                        # expect big round-off errors so just use second_int1
                        second_int = second_int_est
                        error = 'notification: parens is integer, so using n2_est = n_equals - n1'
                else:
                    second_int = second_int_est
                    error = 'notification: n2 estimate only from a single source'
            else:
                first_int = np.nan
                second_int = np.nan
                error = 'error: n1 is neither float nor int (should be impossible)'
        else:
            if type(n1) is int:
                first_int = n1
                if parens is not None:
                    second_int = int(np.round(n1 / parens - n1))
                    error = 'notification: n2 is estimated as int(np.round(n1/parens - n1))'
                else:
                    second_int = np.nan
                    error = 'notification: could not estimate n2 as there are no n_equals or percent'
            elif type(n1) is float:
                first_int = np.nan
                second_int = np.nan
                error = 'error: n_equals is not available, n1 is float'
            else:  # this should never happen
                first_int = np.nan
                second_int = np.nan
                error = 'error: n1 is neither float nor int (should be impossible)'

        if row_male_index is not None:
            return {'male': first_int, 'female': second_int, 'error': error, 'cell_type_from_row': 'male'}
        else:
            return {'male': second_int, 'female': first_int, 'error': error, 'cell_type_from_row': 'female'}
    else:
        return {'male': np.nan, 'female': np.nan, 'error': 'error: neither male nor female are in the row header',
                'cell_type_from_row': None}

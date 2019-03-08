from itertools import groupby

import numpy as np

from pubmedextract.sex_utils.regex_utils import categorize_cell_string


def subdivide(table):
    """
    - Categorize each cell as string, value, or empty
    - Figure out which of the top rows are column headers -> combine them
    - Figure out which of the leftmost columns are row headers -> combine them
    - Put the remaining subtable into a numpy array
    TODO: Common problem: "n (%)" columns are often split up by Omnipage!
          If two adjacent columns have column headers that end with 'n' and '%'/'(%)' respectively,
          then they should be concatenated
    """
    # first, categorize each cell
    table_categories = np.zeros((table.nrow, table.ncol), dtype=np.unicode_)
    for i in range(table.nrow):
        for j in range(table.ncol):
            table_categories[i, j] = categorize_cell_string(table[i, j])

    # figure out how many of the top rows are column headers
    column_header_rows = []
    for i in range(0, table.nrow):
        # sometimes the caption gets lobbed into the first column
        # and splayed across many rows. detect that here:
        all_rows_flag = (table[i, 0].indices[-1][1] + 1 == table.ncol)
        # check if the number of strings is more than 2/3s of the entire row
        s_count = np.sum(table_categories[i, :] == 'S')
        v_count = np.sum(table_categories[i, :] == 'V')
        if all_rows_flag or _row_or_col_is_header(s_count, v_count):
            column_header_rows.append(i)
        else:
            break  # as soon as this is false, we quit
            # TODO: maybe find other rows that are not contiguous with the top rows?

    # figure out how many of the leftmost columns are row headers
    # excluding rows with column headers
    first_non_header_row_ind = _get_and_increment_last(column_header_rows)

    row_header_columns = []
    for i in range(0, table.ncol):
        s_count = np.sum(table_categories[first_non_header_row_ind:, i] == 'S')
        v_count = np.sum(table_categories[first_non_header_row_ind:, i] == 'V')
        # TODO: maybe have a different condition because we have cut out some rows
        if _row_or_col_is_header(s_count, v_count):
            row_header_columns.append(i)
        else:
            break
            # TODO: maybe find other columns that are not contiguous with the top columns?

    # get headers
    column_headers = _combine_omnipage_cell_list(table, column_header_rows, row_flag=True)
    row_headers = _combine_omnipage_cell_list(table, row_header_columns, row_flag=False)

    # edge case if there are no column header rows
    if len(column_headers) == 0:
        column_headers = ['col_' + str(i) for i in range(table.ncol)]

    # get numerical_subtable
    first_non_header_col_ind = _get_and_increment_last(row_header_columns)

    numerical_columns = []
    for col in range(first_non_header_col_ind, table.ncol):
        # extract the part of the column that isn't the header
        col = [str(i) for i in table[:, col]][first_non_header_row_ind:]
        numerical_columns.append(col)

    # we only care about the rows/columns that span the numerical subtable
    column_headers = column_headers[first_non_header_col_ind:]
    row_headers = row_headers[first_non_header_row_ind:]

    # merge columns to previous one if the column is mostly empty
    empty_cols = (table_categories == 'E').mean(0)[first_non_header_col_ind:]
    empty_col_inds = np.where(empty_cols > 0.9)[0]
    ind_ranges_to_merge = [[i - 1, i] for i in empty_col_inds if i > 0]

    # merge columns if they have the same headers
    i = 0
    for k, g in groupby(column_headers):
        g = list(g)
        ind_ranges_to_merge.append(list(range(i, i + len(g))))
        i += len(g)

    # combine overlapping merging index ranges
    ind_ranges_to_merge = _combine_ind_ranges(ind_ranges_to_merge)

    # perform the merge
    # note: only merge the cell contents if they are not identical
    numerical_columns_merged = []
    column_headers_merged = []
    for ind_range_to_merge in ind_ranges_to_merge:
        subcols = [numerical_columns[i] for i in ind_range_to_merge]
        merged_cols = [' '.join(_unique_sorted(j)).strip() for j in zip(*subcols)]
        numerical_columns_merged.append(merged_cols)
        column_headers_merged.append(column_headers[ind_range_to_merge[0]])

    numerical_subtable = np.array(numerical_columns_merged).T

    # if rows of the numerical subtable are all empty
    # then this row's header can be appended to all the subsequent row headers
    # until the next empty set of rows
    # also sometimes there are no row headers, so we have to ensure the lens match
    if len(numerical_subtable) > 1 and len(numerical_subtable) == len(row_headers):
        row_headers, numerical_subtable = _append_row_header_to_subsequent_rows(row_headers, numerical_subtable)

    return column_headers_merged, row_headers, numerical_subtable


def _combine_omnipage_cell_list(table, inds, row_flag):
    """
    Utility function for subdivide
    """
    if row_flag:
        row_or_col_list = [table[i, :] for i in inds]
    else:
        row_or_col_list = [table[:, i] for i in inds]
    return [' '.join(_unique_sorted([str(k) for k in j])).strip()
            for j in zip(*row_or_col_list)]


def _get_and_increment_last(l):
    """
    Utility function for subdivide
    """
    if len(l) > 0:
        return l[-1] + 1
    else:
        return 0


def _row_or_col_is_header(s_count, v_count):
    """
    Utility function for subdivide

    Heuristic for whether a row/col is a header or not.
    """
    if s_count == 1 and v_count == 1:
        return False
    else:
        return (s_count + 1) / (v_count + s_count + 1) >= 2. / 3.


def _combine_ind_ranges(ind_ranges_to_merge):
    """
    Utility function for subdivide

    Function that combines overlapping integer ranges.
    Example
    [[1,2,3], [2,3], [3], [4,5], [5]] -> [[1,2,3], [4,5]]
    """
    ind_ranges_to_merge = sorted(ind_ranges_to_merge)
    stack = []
    result = []
    for curr in ind_ranges_to_merge:
        if len(stack) == 0:
            stack.append(curr)
        elif stack[-1][-1] >= curr[0]:
            prev = stack.pop()
            merged = sorted(list(set(prev + curr)))
            stack.append(merged)
        else:
            prev = stack.pop()
            result.append(prev)
            stack.append(curr)
    result += stack
    return result


def _unique_sorted(seq):
    """
    Utility function for subdivide

    Keeps unique values but preserves order
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def _append_row_header_to_subsequent_rows(row_headers, numerical_subtable):
    """
    Utility function for subdivide

    Some rows headers actually apply to subsequent rows.
    E.g.:

    Sex     np.nan  np.nan
    Male    50      30
    Female  30      20

    For this case, the strong 'Sex' is pre-pended to 'Male' and 'Female' to get:

    Sex - Male    50      30
    Sex - Female  30      20
    """
    empty_flag = (numerical_subtable == '').mean(1) == 1
    empty_rows = list(np.where(empty_flag)[0])
    non_empty_rows = np.where(~empty_flag)[0]
    if len(empty_rows) > 0:
        if empty_rows[-1] != len(row_headers):
            empty_rows.append(len(row_headers))
        all_append_rows = [list(range(empty_rows[i] + 1, empty_rows[i + 1])) for i in range(len(empty_rows) - 1)]
        for i, append_rows in zip(empty_rows, all_append_rows):
            for append_row in append_rows:
                row_headers[append_row] = row_headers[i] + ' - ' + row_headers[append_row]
        row_headers = [row_headers[i] for i in non_empty_rows]
        numerical_subtable = numerical_subtable[non_empty_rows]
    return row_headers, numerical_subtable

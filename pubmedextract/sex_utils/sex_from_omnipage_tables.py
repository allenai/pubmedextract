import numpy as np

acceptable_notifications_ordered = [
    'notification: all cols added up to another one and there is a single overall_col',
    'notification: all cols added up to another one and there is NOT a single overall_col',
    'notification: all cols added up to smaller than biggest overall_col',
    'notification: there were multiple overall_col and they do add up to one of them'
]


acceptable_notifications_unordered = [
    'notification: n_cols = 1, added up all cols',
    'notification: n_cols = 2, added up all cols',
    'notification: n_cols > 2, added up all cols',
    'notification: all cols added up within 10 of biggest overall_col',
    'notification: all cols added up to smaller than biggest single column'
]


unacceptable_notifications = [
    'error: n_cols = 0, columns dropped',
    'error: there were multiple overall_col but they do NOT add up to one of them',
    'error: there were multiple overall_col but they do NOT add up to one of them, columns dropped',
    'notification: all cols added up to smaller than biggest single column, columns dropped',
    'notification: n_cols = 2, added up all cols, columns dropped',
    'notification: n_cols = 1, added up all cols, columns dropped',
    'notification: n_cols > 2, added up all cols, columns dropped',
    'notification: all cols added up within 10 of biggest overall_col, columns dropped'
]


def extract_male_female_counts_from_tables(parsed_tables):
    """
    Gets male and female counts given parsed tables

    First, male and female counts are extracted from every table.

    Then the table with the "best" notification is used as the
    single source of counts for the entire paper. The "best" notification
    is determined by being closest to the top of the list in the
    `acceptable_notifications`.
    """
    results_all = []
    for i, table in enumerate(parsed_tables):
        if table is not None:
            results, _, notification = extract_male_female_counts_from_table(table)
            # we like some notifications better than others:
            # those which indicate all the columns add up to another one
            if notification in acceptable_notifications_ordered:
                results_all.append((acceptable_notifications_ordered.index(notification), results, notification))
            # the rest of the acceptable notifications are all equally good/bad
            # so we care about the table order instead: earlier tables are better than later ones
            # in addition: the ordered acceptable notifications are all better than the unordered ones
            elif notification in acceptable_notifications_unordered:
                ind = len(parsed_tables) + i
                results_all.append((ind, results, notification))

    # sort the results by the notification's index
    if len(results_all) > 0:
        best_result = sorted(results_all, key=lambda x: x[0])[0]
        return best_result[1], best_result[2]
    else:
        return None, 'all errors unacceptable'


def extract_male_female_counts_from_table(table):
    """
    Process all columns and then combine them
    """
    col_headers = np.array(table.col_headers)
    results = np.zeros((2, len(table.col_headers))) * np.nan
    for i in range(len(table.col_headers)):
        r = _process_column(table.counts_matrix[:, :, i], table.row_headers)
        if type(r) is not str:
            results[:, i] = r

    # drop columns if they only have nan results
    good_flag = np.nansum(results, 0) > 0
    col_headers = col_headers[good_flag]
    results = results[:, good_flag]
    ncols = len(col_headers)
    if ncols < len(table.col_headers):
        error_addendum = ', columns dropped'
    else:
        error_addendum = ''

    # there are cases where the 'n' and '%' info is in separate columns
    # so there would be double counting. in this case: take every other column
    # note: it's too risky to do when there are only 2 columns, so make it at least 4
    if results.shape[1] >= 4 and results.shape[1] % 2 == 0 and np.all(np.abs(results[:, ::2] - results[:, 1::2]) <= 2):
        results = results[:, ::2]
        col_headers = col_headers[::2]
        ncols = len(col_headers)

    # check if all columns add up to another column
    biggest = _test_adds_up(results)
    overall_cols = _get_overall_cols(col_headers)
    if ncols > 2:
        if type(biggest) is not str:
            if overall_cols.sum() == 1:
                return results[:, biggest], col_headers[
                    biggest], 'notification: all cols added up to another one and there is a single overall_col'
            else:
                return results[:, biggest], col_headers[
                    biggest], 'notification: all cols added up to another one and there is NOT a single overall_col'
        elif overall_cols.sum() == 1 and biggest == 'too big':
            return results[:, overall_cols], col_headers[
                overall_cols], 'notification: all cols added up to smaller than biggest overall_col'
        elif overall_cols.sum() == 1 and biggest == 'within 10':
            return results[:, overall_cols], col_headers[
                overall_cols], 'notification: all cols added up within 10 of biggest overall_col'
        elif overall_cols.sum() == 0 and biggest == 'too big':
            return results.sum(
                1), col_headers, 'notification: all cols added up to smaller than biggest single column' + error_addendum
        elif overall_cols.sum() > 1 and biggest == 'too small':
            results_sub, col_headers_sub = results[:, overall_cols], col_headers[overall_cols]
            biggest_sub = _test_adds_up(results_sub)
            if type(biggest_sub) is not str:
                return results_sub[:, biggest_sub], col_headers_sub[
                    biggest_sub], 'notification: there were multiple overall_col and they do add up to one of them' + error_addendum
            else:
                return None, None, 'error: there were multiple overall_col but they do NOT add up to one of them' + error_addendum
        else:
            return results.sum(1), col_headers, 'notification: n_cols > 2, added up all cols' + error_addendum
    elif ncols == 2:
        return results.sum(1), col_headers, 'notification: n_cols = 2, added up all cols' + error_addendum
    elif ncols == 1:
        return results.sum(1), col_headers, 'notification: n_cols = 1, added up all cols' + error_addendum
    else:
        return None, None, 'error: n_cols = 0' + error_addendum


def _process_column(col, row_headers):
    """
    Utility function for extract_male_female_counts_from_table

    Process multiple rows of a single column.
    TODO: sometimes (k-1) ROWS add up to the k-th largest ROW.
    e.g. https://www.semanticscholar.org/paper/Rivaroxaban-versus-enoxaparin%2Fvitamin-K-antagonist-Bauersachs-Lensing/364d8c900f0b8dbac677800d63710cd73f2f5135/figure/0
    """

    def process_one_row_of_col(subcol):
        nans = np.isnan(subcol)
        counts = subcol[0, :]
        male_exist = ~nans[0] or ~nans[2]
        female_exist = ~nans[1] or ~nans[3]
        if not (male_exist and female_exist):
            return 'error(1 row): both male and female not available'
        elif nans.sum() == 0:
            # sums of male and female counts are within 5%
            if np.abs(counts[:2].sum() - counts[2:].sum()) / counts[:2].sum() <= 0.05:
                return counts[:2]
            else:
                return 'error(1 row): counts and estimated counts do not match'
        elif nans.sum() == 2:
            # take the truth where available and estimate where available
            return np.nansum(subcol.reshape((2, 2)), axis=0)
        else:
            return 'error(1 row): estimate exists for only 1 of the 2 sexes'

    def process_two_rows_of_col(subcol):
        nans = np.isnan(subcol)
        nans_per_type = nans.sum(axis=1)
        # there should only be at most a single non-nan across the rows
        if np.all(nans_per_type > 0):
            counts = np.nansum(subcol, 1)
            counts_exist = np.all(nans_per_type[:2] < 2)
            ests_exist = np.all(nans_per_type[2:] < 2)
            if counts_exist and ests_exist:
                if np.abs(counts[:2].sum() - counts[2:].sum()) / counts[:2].sum() <= 0.05:
                    return counts[:2]
                else:
                    return 'error(2 rows): counts and estimated counts do not match'
            elif counts_exist:
                return counts[:2]
            elif ests_exist:
                return 'error(2 rows): estimates exist but counts are not sufficient'  # this should never happen
            else:
                return 'error(2 rows): neither counts nor ests are sufficient'
        else:
            return 'error(2 rows): more than a single non-nan value per row'

    n_rows = col.shape[1]
    if n_rows == 1:
        return process_one_row_of_col(col)
    elif n_rows == 2:
        return process_two_rows_of_col(col)
    elif n_rows > 20:
        return ('error(n rows): too many rows')
    else:
        # if all rows have the same row header
        # then sum all of their counts together
        if len(set([tuple(i) for i in row_headers])) == 1:
            results = np.array([0., 0.])
            for i in range(0, n_rows):
                subresult = process_one_row_of_col(col[:, i:i + 1])
                if type(subresult) is not str:
                    results += subresult
                else:
                    return 'error(n rows): ' + subresult
            return results
        # here we're assuming consecutive pairs yield reasonable results
        # for this, there must be an even number of rows
        elif n_rows % 2 == 0:
            results = np.array([0., 0.])
            for i in range(0, n_rows - 1, 2):
                subresult = process_two_rows_of_col(col[:, i:i + 2])
                if type(subresult) is not str:
                    results += subresult
                else:
                    return 'error(n rows), ' + subresult
            return results
        else:
            return 'error(n rows): other reasons'


def _test_adds_up(X, axis=1):
    """
    Utility function for extract_male_female_counts_from_table

    Check if the biggest row/col is the sum of the rest
    """
    if axis == 0:
        X = X.T
    if X.shape[1] == 0:
        return None
    biggest = np.argmax(X.sum(0))
    Xsub = X[:, np.delete(np.arange(0, X.shape[1]), biggest)]
    Xsubsum = Xsub.sum(1)
    Xbiggest = X[:, biggest]
    diffs = Xbiggest - Xsubsum
    # if any of male OR female diffs are very close
    # and all are greater or equal to 0
    # note: using np.any instead of np.all because
    # often either male or female is estimated and there
    # is a larger margin of error
    if np.any(np.abs(diffs) == 0) and np.all(diffs >= 0):
        return biggest
    elif np.all(diffs < -1):
        return 'too small'
    elif np.all(diffs > 1):
        return 'too big'
    elif np.all(np.abs(diffs) <= 10):
        return 'within 10'
    else:
        return 'neither'


def _get_overall_cols(col_headers):
    """
    Utility function for extract_male_female_counts_from_table

    Checks if column is likely to be a 'total' column
    """

    def has_overall(col_header):
        flag = (
            'all' in col_header or
            'full' in col_header or
            'pooled' in col_header or
            'total' in col_header or
            'overall' in col_header or
            'combined' in col_header or
            ('sample' in col_header and 'size' in col_header)
        )
        return flag

    return np.array([has_overall(i) if len(i) > 0 else False for i in col_headers])

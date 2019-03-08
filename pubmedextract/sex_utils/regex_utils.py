import re

import numpy as np

from pubmedextract.sex_utils.namedtuples import NumericalCell

"""
Regexes
"""


# characters that are common in numerical fields
regex_num_charset = re.compile('[\d±%=.<>,–\-(){}\[\]]')

# commas not followed by a space: 3,000 -> 3000
regex_comma_space = re.compile(r'(\d{1,3}(?:,\d{3})+)')

# characters that should be removed everywhere for ease of life
regex_remove_chars = re.compile('[*;‡†":]')

# find 'n=' or 'n =' or 'n = '
regex_n_equals = re.compile("([𝑛nN]\s*=\s?)(\d+)")

# common pattern #1: 652 or 14.0% (negative numbers are not supported)
regex_single_number = re.compile('^(?P<n1>[\d.%]+)$')  # print(regex_single_number.findall('124.5%'))

# common pattern #2: 215/4732
regex_slash = re.compile('^(?P<n1>[\d.%]+)/(?P<n2>[\d.%]+)$')  # print(regex_slash.findall('63/2'))

# common pattern #3: 215 (12.5%)
regex_number_parens = re.compile(
    '^(?P<n1>[\d.%]+)\s*\((?P<parens>[\d.%\s]*?)\)$')  # regex_integer_parens.findall('320 (12.5%)')

# common pattern #4: 215/456( 12.5% )
regex_slash_parens = re.compile(
    '^(?P<n1>[\d.%]+)/(?P<n2>[\d.%]+)\s*\((?P<parens>[\d.%\s]*?)\)$')  # regex_slash_parens.findall('320/360 (12.5%)')


'''
Regex-heavy functions for cell and table parsing
'''


def regex_numerical_field(s, row_header, col_header, n_equals_row, n_equals_col):
    """
    Parse a string s to see if it is one of four common patterns
    that occur in medical table participant counts (any value can be float and/or have a %):

    Pattern 1: 30%
    Pattern 2: 50.5/90.2
    Pattern 3: 30 (40%)
    Pattern 4: 20/40 (60)

    Returns NumericalCell(n1=n1, n2=n2, parens=parens)
            where `n1` is the first value found in a field,
            `n2` is the second value found and `parens` is the value found in the parenthesis,
            which is usually a percentage
    """
    s1 = s.strip().lower()
    # remove commas in digits such as 1,209
    for i in regex_comma_space.findall(s1):
        s1 = s1.replace(i, i.replace(',', ''))
    # remove characters that commonly occur in numerical fields but mess up parsing
    s2 = regex_remove_chars.sub(' ', s1)
    # replace e.g. 'n = 30' with '30'
    s3 = regex_n_equals.sub(r'\2', s2)

    # try to match common formats
    s_final = s3.strip()
    match_single_number = regex_single_number.search(s_final)
    match_slash = regex_slash.search(s_final)
    match_parens = regex_number_parens.search(s_final)
    match_slash_parens = regex_slash_parens.search(s_final)
    match_combined = match_single_number or match_slash or match_parens or match_slash_parens

    # extract results
    if match_combined:  # first number is always extracted
        n1, n1_is_percent = format_num(match_combined.group('n1'))
    else:
        n1, n1_is_percent = None, None

    if match_slash or match_slash_parens:  # m2 and m4 have second number
        n2, n2_is_percent = format_num(match_combined.group('n2'))
    else:
        n2, n2_is_percent = None, None

    if match_parens or match_slash_parens:  # m3 and m4 have parens content
        parens, parens_is_percent = format_num(match_combined.group('parens'))
    else:
        parens, parens_is_percent = None, None

    # some edge cases that can be fixed here
    n_equals = n_equals_row or n_equals_col
    percent_before_n = (row_header.index('%') if '%' in row_header else len(row_header)) < (
        row_header.index('n') if 'n' in row_header else -1)
    if n1 is not None and parens is not None and n2 is None:
        # edge case: 50% (30) -> switch them around to be 30 (50%)
        if (n1_is_percent and not parens_is_percent) or percent_before_n:
            n1, parens = parens, n1
        # edge case: n (x%) is flipped. we can guess this is the case if both '%' and 'n' are in row in a certain order
        # or, if n_equals is around, the numbers work out correctly
        elif (n_equals is not None and
              np.abs(n_equals * n1 / 100 - parens) < 5 and
              not np.abs(n_equals * parens / 100 - n1) < 5):
            n1, parens = parens, n1
    # edge case: 30/50% -> stick second number into parens location to be 30 (50%)
    elif n1 is not None and not n1_is_percent and n2 is not None and n2_is_percent and parens is None:
        parens = n2
        n2 = None

    # a way to tell if n1 and n2 are probably %
    row_header_has_no_n = 'n' not in row_header and 'no' not in row_header and '𝑛' not in row_header
    header_has_percent = '%' in row_header or '%' in col_header
    if parens is None and row_header_has_no_n and header_has_percent:
        if type(n1) is int and 0 <= n1 <= 100:
            n1 *= 1.0

        if type(n2) is int and 0 <= n2 <= 100:
            n2 *= 1.0

    # simple check if parens is a float
    if n1 is not None and type(parens) is int and ('%' in row_header or '%' in col_header):
        parens *= 1.0

    # return results if there is at least one value
    if n1 is None and n2 is None and parens is None:
        return None
    else:
        return NumericalCell(n1=n1, n2=n2, parens=parens)


def extract_n_equals_value(s):
    """
    Extracts the value that follows N =, e.g. N = 35
    """
    # remove commas
    s1 = s.strip().lower()
    for i in regex_comma_space.findall(s1):
        s1 = s1.replace(i, i.replace(',', ''))
    # find n =
    m = regex_n_equals.findall(s1)
    # there should only be exactly one 'n = '
    if len(m) == 1:
        return int(m[0][1])
    else:
        return None


def categorize_cell_string(s):
    """
    Categorizes each cell string into Empty, String or Value
    """
    # special case: cells with 'n = ' are almost always part of the row/col headers
    # special case: where there are numerical chars like '%' but no actual digits anywhere
    s = str(s).strip().lower()
    charset_count = len(regex_num_charset.findall(s))
    digits_count = len(re.findall(r'\d', s))
    has_n_equals = len(regex_n_equals.findall(s))
    # empty is can also be '-'
    if len(s) == 0 or s == '-':
        return 'E'
    elif (has_n_equals or # 'n = 30' usually shows up in header
          digits_count == 0 or # no digits
          charset_count / len(s) < 2 / 3):
        return 'S'
    else:
        return 'V'


def format_num(s):
    """
    Figures out if string s is a int or float
    and whether it's a percentage
    """
    s = s.strip()
    percent = False
    if s.endswith('%'):
        percent = True
        s = s[:-1].strip()
    # later, float will mean percent
    try:
        x = float(s)
        if x.is_integer() and not percent and '.' not in s:
            x = int(x)
    except:
        x = None
    return x, percent

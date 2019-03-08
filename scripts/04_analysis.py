import psycopg2
import pickle
import numpy as np
from collections import defaultdict
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

sns.set(context='poster')
pd.set_option('display.width', 1000) # to display more columns

# load data needed for analysis
with open('data/pubmed_id_mesh_map.pickle', 'rb') as f:
    pmid_to_mesh = pickle.load(f)

with open('data/pubmedextract_results.pickle', 'rb') as f:
    pubmedextract_results = pickle.load(f)

with open('data/pubmed_s2_id_maps.pickle', 'rb') as f:
    s2id_to_pmid, pmid_to_s2id = pickle.load(f)

# import counts_with_disease
category_to_disease_counts = {}
with open('data/disease_category_prevalence_global.tsv', 'r') as f:
    for line in f:
        cat, _, _, _, _, male_count, female_count = line.strip().split('\t')
        male_count = float(male_count.replace(',', ''))
        female_count = float(female_count.replace(',', ''))
        category_to_disease_counts[cat] = {'male_count': male_count, 'female_count': female_count}

category_map = {'Cardiovascular diseases': 'Cardiovascular diseases',
 'Neoplasms (cancer)': 'Neoplasms',
 'Chronic kidney disease': 'Chronic kidney disease',
 'Chronic respiratory diseases': 'Chronic respiratory diseases',
 'Diabetes mellitus': 'Diabetes mellitus',
 'Digestive diseases': 'Digestive diseases',
 'Hepatitis A/B/C/E': 'Hepatitis A, B, C, and E',
 'HIV/AIDS': 'HIV/AIDS',
 'Mental disorders': 'Mental disorders',
 'Musculoskeletal disorders': 'Musculoskeletal disorders',
 'Neurological disorders': 'Neurological disorders',
 'Total': 'Total'}

# get also a Total count
total = {
    'male_count': np.sum([i['male_count'] for i in category_to_disease_counts.values()]),
    'female_count': np.sum([i['female_count'] for i in category_to_disease_counts.values()])
}
category_to_disease_counts['Total'] = total

# get a mapping from mesh term to disease category
mesh_to_category = {}
with open('data/mesh2category_global.tsv', 'r') as f:
    for line in f:
        mesh, cat = line.strip().split('\t')
        if cat != 'None of the above':
            mesh_to_category[mesh] = cat.strip('*')

# get disease counts from AACT
# load from disk data collected at end of 2018
with open('data/aact_query_jan_7_2019.pickle', 'rb') as f:
    sex, diseases = pickle.load(f)

"""
# if you prefer to get more recent data, use the code below instead
# you will first need to create an account here: https://aact.ctti-clinicaltrials.org/users/sign_up
# then update user and password fields
params = {
  'dbname': 'aact',
  'user': 'YOUR_NAME',
  'password': 'YOUR_PASSWORD',
  'host': 'aact-db.ctti-clinicaltrials.org',
  'port': 5432
}

conn = psycopg2.connect(**params)

diseases_query = '''
select c.nct_id, c.mesh_term as mesh 
from browse_conditions as c join studies as d on c.nct_id = d.nct_id
where d.overall_status = 'Completed';
'''
diseases = pd.read_sql_query(diseases_query, conn)

sex_query = '''
select s.study_first_submitted_date, b.nct_id, b.category as cat, sum(b.param_value_num) as total_participants
from
(
    select bm.nct_id, lower(concat(bm.category, bm.classification)) as category, bm.param_value_num, bm.title, bm.param_type
    from baseline_measurements as bm join result_groups as r
    on bm.nct_id = r.nct_id and bm.ctgov_group_code = r.ctgov_group_code
    and lower(r.title) != 'total'
) as b
join studies as s on b.nct_id = s.nct_id
where s.overall_status = 'Completed' 
and s.study_first_submitted_date < '2019-01-01'
and (b.param_type = 'Count of Participants' or b.param_type = 'Number')
and (b.title ~* 'sex' or b.title ~* 'gender') 
and (b.category = 'male' or b.category = 'female')
group by b.nct_id, cat, s.study_first_submitted_date
order by b.nct_id;
'''
sex = pd.read_sql_query(sex_query, conn)

# keep nct_ids that have both male and female
sex = sex.pivot(index='nct_id', columns='cat', values='total_participants')
sex.dropna(axis=0, how='any', inplace=True)
"""


# get male/female counts for each nct_id
nct_id_to_counts = {}
for nct_id, counts in zip(sex.index, sex.values):
    female_counts, male_counts = counts
    counts_dict = {'males': male_counts, 'females': female_counts}
    nct_id_to_counts[nct_id] = counts_dict

# get all the mesh terms for each nct_id
nct_id_to_mesh = defaultdict(list)
for nct_id, mesh in diseases.values:
    if nct_id in nct_id_to_counts:
        nct_id_to_mesh[nct_id].append(mesh)

# assemble aggregate AACT counts
footnote_string = '‡'
# assemble aggregate PubMedExtract counts
category_to_counts_pubmedextract = defaultdict(list)
more_than_zero = 0
exactly_one = 0
more_than_zero_counts = 0
exactly_one_counts = 0
for pmid, counts in pubmedextract_results.items():
    if counts['males'] > 0 and counts['females'] > 0:
        categories = list(set([
            mesh_to_category[i]
            for i in pmid_to_mesh[pmid]
            if i in mesh_to_category
        ]))
        counts['pmid'] = pmid
        counts['s2id'] = pmid_to_s2id[pmid]
        if len(categories) > 0:
            for cat in categories:
                category_to_counts_pubmedextract[cat].append(counts)
            more_than_zero += 1
            more_than_zero_counts += counts['males'] + counts['females']
        if len(categories) == 1:
            exactly_one += 1
            exactly_one_counts += counts['males'] + counts['females']

print('Number of PubMed records with exactly a single category and all records, and diff:', exactly_one, more_than_zero, more_than_zero - exactly_one)
print('Fraction of PubMed records that fall into a single category:', exactly_one / more_than_zero)
print('Total number of subjects:', more_than_zero_counts)
footnote_string += '%d published articles (%d%%) with %d subjects (%d%%) and ' % (
    more_than_zero - exactly_one,
    round(100 * (1 - exactly_one / more_than_zero)),
    more_than_zero_counts - exactly_one_counts,
    round(100 * (1 - exactly_one_counts / more_than_zero_counts))
)

# also make a total category that contains sums of all others
category_to_counts_pubmedextract['Total'] = sum(category_to_counts_pubmedextract.values(), [])

print('Total number of papers for which we have extracted counts:', len(pubmedextract_results))
print('Number of papers that fall into a predefined medical category:', more_than_zero)
print('\n--------------------------------------------------------------------')
print('Paper counts per category (some papers have more than one category):')
print('--------------------------------------------------------------------')
for cat, counts in sorted(category_to_counts_pubmedextract.items()):
    print(len(counts), category_map[cat], sep='\t')


category_to_counts_aact = defaultdict(list)
more_than_zero = 0
exactly_one = 0
more_than_zero_counts = 0
exactly_one_counts = 0
for nct_id, counts in nct_id_to_counts.items():
    if counts['males'] > 0 and counts['females'] > 0:
        categories = list(set([
            mesh_to_category[i]
            for i in nct_id_to_mesh[nct_id]
            if i in mesh_to_category
        ]))
        counts['nct_id'] = nct_id
        if len(categories) > 0:
            for cat in categories:
                category_to_counts_aact[cat].append(counts)
            more_than_zero += 1
            more_than_zero_counts += int(counts['males']) + int(counts['females'])
        if len(categories) == 1:
            exactly_one += 1
            exactly_one_counts += int(counts['males']) + int(counts['females'])

print('Number of AACT records with exactly a single category and all records, and diff:', exactly_one, more_than_zero, more_than_zero - exactly_one)
print('Fraction of AACT records that fall into a single category:', exactly_one / more_than_zero)
print('Total number of subjects:', more_than_zero_counts)
footnote_string += '%d AACT records (%d%%) with %d subjects (%d%%) contributed to sex bias estimates for > 1 disease category.' % (
    more_than_zero - exactly_one,
    round(100 * (1 - exactly_one / more_than_zero)),
    more_than_zero_counts - exactly_one_counts,
    round(100 * (1 - exactly_one_counts / more_than_zero_counts))
)

# also make a total category that contains sums of all others
category_to_counts_aact['Total'] = sum(category_to_counts_aact.values(), [])

print('Total number of AACT trials for which we have counts:', len(nct_id_to_counts))
print('Number of AACT trials that fall into a predefined medical category:', more_than_zero)
print('\n--------------------------------------------------------------------')
print('AACT trials per category (some papers have more than one category):')
print('--------------------------------------------------------------------')
for cat, counts in sorted(category_to_counts_aact.items()):
    print(len(counts), category_map[cat], sep='\t')

print(footnote_string)


'''
The code below generates the two main results tables as LaTeX code
'''
def get_bootstrap_samples(list_of_arrays, index_array):
    inds_bootstrap = np.random.choice(index_array, size=len(index_array), replace=True)
    return np.array([i[inds_bootstrap] for i in list_of_arrays])


def get_ci_string(odds_ratio_middle, odds_ratio_lower, odds_ratio_upper):
    string = '%2.2f (%2.2f, %2.2f)' % (np.round(odds_ratio_middle, 2),
                                       np.round(odds_ratio_lower, 2),
                                       np.round(odds_ratio_upper, 2))
    return string.replace('-', '−')


def compute_bootstrap_pvalue(bootstrapped_statistics, null_mean=0, ci_alpha=0.05):
    """ONLY works for two sided tests"""

    # generate grid of possible CI confidence levels
    alphas = np.linspace(start=0.00001, stop=1.0, num=100000)

    # for each alpha, compute the Lower and Upper CI bounds and check if null mean is contained within
    lowers = np.percentile(bootstrapped_statistics, 100 * alphas / 2)
    uppers = np.percentile(bootstrapped_statistics, 100 * (1 - alphas / 2))
    decisions = (lowers <= null_mean) & (null_mean <= uppers)  # true means accept

    # also compute the CI for our desired alpha to report
    lower_CI = np.percentile(bootstrapped_statistics, 100 * ci_alpha / 2)
    upper_CI = np.percentile(bootstrapped_statistics, 100 * (1 - ci_alpha / 2))
    middle_CI = np.mean(bootstrapped_statistics)

    # find the largest alpha that led to an 'accept' decision (or smallest alpha that lead to 'reject')
    # e.g. large alpha 0.2 is an 80% CI which is narrow --> likely to reject
    #      small alpha 0.0001 is a 99.99% CI which is wide --> likely to accept
    index_accepts = [i for i, decision in enumerate(decisions) if decision]
    if len(index_accepts) == 0:
         pval_string = '0.00001'
    else:
        pvalue = np.max(alphas[index_accepts])
        if pvalue > 0.001:
            pval_string = 'NS'
        else:
            pval_string = '%2.4f' % pvalue
    return middle_CI, lower_CI, upper_CI, pval_string

def get_bootstrap_cis(x,
                      prevalence_counts,
                      bootstrap_n=1000,
                      ci_alpha=0.05,
                      year_lower=None,
                      year_upper=None,
                      paper_as_unit_flag=True,
                      odds_ratio_flag=False):
    '''
    paper_as_unit_flag=True means that we weigh each paper equally
    paper_as_unit_flag=False means that we weigh each paper proportional to its number of participants
    odds_ratio_flag=True means we use odds ratio
    odds_ratio_flag=False means we use prevalence fraction

    '''
    # extract data into a useful format
    f = np.array([i['females'] for i in x])
    m = np.array([i['males'] for i in x])

    if 'year' in x[0]:
        years = np.array([i['year'] for i in x])

        # sub on year
        if year_lower is not None:
            flag = years >= year_lower
            f = f[flag]
            m = m[flag]
            years = years[flag]
        if year_upper is not None:
            flag = years <= year_upper
            f = f[flag]
            m = m[flag]
            years = years[flag]

    # exclude trials where there are 0 men or 0 women - single-sex isn't as interesting
    flag = (f > 0) & (m > 0)
    f = f[flag]
    m = m[flag]
    if 'year' in x[0]:
        years = years[flag]

    # for odds_ratio_flag=True and paper_as_unit_flag=True
    f_enroll_frac = f / prevalence_counts['female_count']
    m_enroll_frac = m / prevalence_counts['male_count']
    odds_ratio = np.log(m_enroll_frac / f_enroll_frac)

    # for odds_ratio_flag=False and paper_as_unit_flag=True
    f_true_prev = prevalence_counts['female_count'] / (
                prevalence_counts['female_count'] + prevalence_counts['male_count'])
    f_prev_frac = f / (f + m)
    m_prev_frac = m / (f + m)

    # total subjects
    n_subj = int(np.sum(f + m))
    n_study = len(f)

    if paper_as_unit_flag:
        n = n_study
    else:
        n = n_subj

    # for the bootstrap function
    index_array = np.arange(len(f))

    # do bootstrap
    f_means = []
    m_means = []
    odds_ratio_means = []
    f_prev_frac_means = []
    for i in range(bootstrap_n):
        if odds_ratio_flag:
            if paper_as_unit_flag:
                f_enroll_frac_boot, m_enroll_frac_boot, odds_ratio_boot = get_bootstrap_samples(
                    [f_enroll_frac, m_enroll_frac, odds_ratio], index_array)
                f_means.append(f_enroll_frac_boot.mean())
                m_means.append(m_enroll_frac_boot.mean())
                odds_ratio_means.append(odds_ratio_boot.mean())
            else:
                f_boot, m_boot = get_bootstrap_samples([f, m], index_array)
                f_enroll = np.sum(f_boot) / prevalence_counts['female_count']
                m_enroll = np.sum(m_boot) / prevalence_counts['male_count']
                f_means.append(f_enroll)
                m_means.append(m_enroll)
                odds_ratio_means.append(np.log(m_enroll / f_enroll))
        else:
            if paper_as_unit_flag:
                f_prev_frac_boot, m_prev_frac_boot = get_bootstrap_samples([f_prev_frac, m_prev_frac], index_array)
                f_means.append(np.mean(f_prev_frac_boot))
                m_means.append(np.mean(m_prev_frac_boot))
                f_prev_frac_means.append(np.mean(f_prev_frac_boot) - f_true_prev)
            else:
                f_boot, m_boot = get_bootstrap_samples([f, m], index_array)
                f_prev_frac_boot = np.sum(f_boot) / (np.sum(f_boot) + np.sum(m_boot))
                m_prev_frac_boot = np.sum(m_boot) / (np.sum(f_boot) + np.sum(m_boot))
                f_means.append(f_prev_frac_boot)
                m_means.append(m_prev_frac_boot)
                f_prev_frac_means.append(f_prev_frac_boot - f_true_prev)

    if odds_ratio_flag:
        f_middle = "{0:.2e}".format(np.mean(f_means))
        m_middle = "{0:.2e}".format(np.mean(m_means))
        odds_ratio_middle, odds_ratio_lower, odds_ratio_upper, p_val = compute_bootstrap_pvalue(odds_ratio_means,
                                                                                                null_mean=1,
                                                                                                ci_alpha=ci_alpha)
        return f_middle, m_middle, np.exp(odds_ratio_middle), np.exp(odds_ratio_lower), np.exp(odds_ratio_upper), n, p_val
    else:
        f_middle = "{:3.2f}".format(np.mean(f_means))
        m_middle = "{:3.2f}".format(np.mean(m_means))
        f_prev_frac_middle, f_prev_frac_lower, f_prev_frac_upper, p_val = compute_bootstrap_pvalue(f_prev_frac_means,
                                                                                                   null_mean=0,
                                                                                                   ci_alpha=ci_alpha)
        return f_middle, m_middle, f_prev_frac_middle, f_prev_frac_lower, f_prev_frac_upper, n, p_val


for paper_as_unit_flag in [True, False]:
    total_papers_pubmedextract = []
    total_papers_pubmedextract_used = []
    total_papers_aact = []
    total_papers_aact_used = []
    if paper_as_unit_flag:
        print('''
       Disease Category	Fprev	No. of Published Articles	Fsubj	Sex Bias (95% CI)	P ≤	No. of Clinical Trial Records	Fsubj	Sex Bias (95% CI)	P ≤
        ''')
    else:
        print('''
        Disease Category	Fprev	No. of Subjects	Fsubj	Sex Bias (95% CI)	P ≤	No. of Subjects	Fsubj	Sex Bias (95% CI)	P ≤
         ''')
    for category, prevalence_counts in category_to_disease_counts.items():
        if category in category_to_counts_pubmedextract and category in category_to_counts_aact:
            total_papers_pubmedextract.append(len(category_to_counts_pubmedextract[category]))
            total_papers_aact.append(len(category_to_counts_aact[category]))

            # female prevalence fraction (global)
            FPrF_global = prevalence_counts['female_count'] / (
                        prevalence_counts['male_count'] + prevalence_counts['female_count'])

            # pubmedextract
            f_enroll_frac_middle, m_enroll_frac_middle, bias_middle, bias_lower, bias_upper, n, p_val = get_bootstrap_cis(
                category_to_counts_pubmedextract[category],
                prevalence_counts,
                paper_as_unit_flag=paper_as_unit_flag)
            total_papers_pubmedextract_used.append(n)
            bias_str = get_ci_string(bias_middle, bias_lower, bias_upper)
            pubmedextract_str = '%s \t %s \t %s \t %s' % (
                "{:,}".format(n),
                f_enroll_frac_middle,
                bias_str,
                p_val
            )

            # aact
            f_enroll_frac_middle, m_enroll_frac_middle, bias_middle, bias_lower, bias_upper, n, p_val = get_bootstrap_cis(
                category_to_counts_aact[category],
                prevalence_counts,
                paper_as_unit_flag=paper_as_unit_flag)
            total_papers_aact_used.append(n)
            bias_str = get_ci_string(bias_middle, bias_lower, bias_upper)
            aact_str = '%s \t %s \t %s \t %s' % (
                "{:,}".format(n),
                f_enroll_frac_middle,
                bias_str,
                p_val
            )

            print('{:35s} \t {:2.2f} \t {:35s} \t {:35s}'.format(
                category_map[category],
                FPrF_global,
                pubmedextract_str,
                aact_str
            ))

    print('\n\n')

'''
The code below generates the results-split-by-time tables as LaTeX code
'''
def get_string(odds_ratio_middle, n, p_val):
    string = '%s \t %2.2f' %("{:,}".format(n), np.round(odds_ratio_middle, 2))
    if p_val != 'NS':
        string += '†'
    return string.replace('-', '−')


years_lower = [None, 1994, 1999, 2004, 2009, 2014, None]
years_upper = [1993, 1998, 2003, 2008, 2013, 2018, 2018]

for paper_as_unit_flag in [True, False]:

    print('''
    Disease Category \t <= 1993 \t 1994 - 1998 \t 1999 - 2003 \t 2004 - 2008 \t 2009 - 2013 \t 2014 - 2018 \t Total ''')

    for category, prevalence_counts in category_to_disease_counts.items():
        if category in category_to_counts_pubmedextract and category in category_to_counts_aact:
            row_str = ''
            for year_lower, year_upper in zip(years_lower, years_upper):
                _, _, middle, lower, upper, n, p_val = get_bootstrap_cis(category_to_counts_pubmedextract[category],
                                                                         prevalence_counts,
                                                                         year_lower=year_lower,
                                                                         year_upper=year_upper,
                                                                         paper_as_unit_flag=paper_as_unit_flag)
                year_str = get_string(middle, n, p_val)
                row_str += '\t' + year_str
            print('{:30s}{:80s}'.format(
                category_map[category],
                row_str
            ))


'''
The code below generates the scatterplot figure and saves it to the data directory
'''
category = 'Cardiovascular diseases'
counts = category_to_counts_aact[category]
prevalence_counts = category_to_disease_counts[category]

f_true_prev = prevalence_counts['female_count'] / (prevalence_counts['female_count'] + prevalence_counts['male_count'])

# extract data into a useful format
f = np.array([i['females'] for i in counts])
m = np.array([i['males'] for i in counts])

# exclude trials where there are 0 men or 0 women - single-sex isn't as interesting
flag = (f > 0) & (m > 0)
f = f[flag]
m = m[flag]
n = f + m
bias = f / (f + m) - f_true_prev

rgba_colors = np.zeros((len(n), 4))
rgba_colors[:, 0] = 1.0
rgba_colors[:, 3] = n/np.max(n)

plt.figure(figsize=(36, 18))

plt.subplot(1, 2, 1)
plt.scatter(n, bias, s=250, alpha=0.2)
plt.ylim([-0.6, 0.6])
plt.title('Trials with Equal Weight', fontsize=60)
plt.xscale('log')
plt.xlabel('Number of participants in trial', fontsize=55)
plt.ylabel('Sex bias', fontsize=55)
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)

plt.subplot(1, 2, 2)
plt.scatter(n, bias, s=250, color=rgba_colors)
plt.ylim([-0.6, 0.6])
plt.title('Trials Weighted by Participants', fontsize=60)
plt.xscale('log')
plt.xlabel('Number of participants in trial', fontsize=55)
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)

plt.tight_layout()
plt.savefig('data/trial_n_vs_bias.png', dpi=60, quality=100)


'''
The code below is analysis via fixed-effects models of the relationship between study size and per-publication bias 
'''
all_years = []
all_n_subjects = []
all_bias = []
all_categories = []
for category, prevalence_counts in category_to_disease_counts.items():
    if category in category_to_counts_pubmedextract and category in category_to_counts_aact and category != 'Total':
        x = category_to_counts_pubmedextract[category]

        # np is better format for this
        f = np.array([i['females'] for i in x])
        m = np.array([i['males'] for i in x])
        years = np.array([i['year'] for i in x])

        # non-single-sex trials only
        flag = (f > 0) & (m > 0)
        f = f[flag]
        m = m[flag]
        years = years[flag]

        # compute bias
        f_true_prev = prevalence_counts['female_count'] / (
                prevalence_counts['female_count'] + prevalence_counts['male_count'])
        n_subjects = f + m
        bias = f / n_subjects - f_true_prev

        # append to store
        all_years.extend(list((years - 1966)/(2018 - 1966)))
        all_n_subjects.extend(list(n_subjects))
        all_bias.extend(list(bias))
        all_categories.extend([category] * len(years))

data = pd.DataFrame({'year': all_years,
                     'n_subjects': all_n_subjects,
                     'bias': all_bias,
                     'diseasese_category': all_categories})

# we will use categorical deciles for number of subjects
data['n_subjects_decile'] = pd.qcut(data['n_subjects'], 10)

model = smf.glm(formula="bias ~ year + C(diseasese_category) + C(n_subjects_decile)", data=data).fit()
print(model.summary())

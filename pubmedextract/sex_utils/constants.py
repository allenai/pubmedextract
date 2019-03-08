'''
Sex/gender
'''


SEX_SYNONYMS = {'Gender', 'Sex/Gender, Customized', 'Sex: Female, Male', 'Gender, Customized'}

INVERSE_SEX_NGRAM_MAP = {
    'Male': [
        ['male'],
        ['males'],
        ['men']
    ],
    'Female': [
        ['female'],
        ['females'],
        ['women']
    ]
}

SEX_VOCAB = {'female', 'females', 'male', 'males', 'men', 'women'}


'''
Race/ethnicity
'''

CANONICAL_RACES = {
    'American Indian or Alaska Native',
    'Asian',
    'Black or African American',
    'Hispanic or Latino',
    'More than one race',
    'Native Hawaiian or Other Pacific Islander',
    'Not Hispanic or Latino',
    'Other',
    'Unknown or Not Reported',
    'White'
}

RACE_SYNONYMS = {
    'Ethnicity (NIH/OMB)',
    'Black Race',
    'White race',
    'Race/Ethnicity',
    'Ethnicity',
    'Race-Ethnicity',
    'Predominant Race',
    'Race',
    'Donor Race/Ethnicity',
    'Race/Ethnicity, Customized',
    'Black race',
    'Race (NIH/OMB)'
}

RACE_CANONICAL_MAP = {
    'Mixed Race': 'More than one race',
    'East Asian': 'Asian',
    'Asian - East Asian Heritage': 'Asian',
    'Asian - South East Asian Heritage': 'Asian',
    'Asian - Japanese Heritage': 'Asian',
    'Hispanic': 'Hispanic or Latino',
    'Latino': 'Hispanic or Latino',
    'Black': 'Black or African American',
    'African American': 'Black or African American',
    'African American/African Heritage': 'Black or African American',
    'African-American': 'Black or African American',
    'Black/African American': 'Black or African American',
    'Caucasian': 'White',
    'White - White/Caucasian/European Heritage': 'White',
    'White or Caucasian': 'White',
    'Native American': 'American Indian or Alaska Native',
    'Missing': 'Unknown or Not Reported',
    'Unknown': 'Unknown or Not Reported',
}

INVERSE_RACE_NGRAM_MAP = {
    'American Indian or Alaska Native': [
        ['native', 'american'],
        ['native', 'americans'],
        ['american', 'indian'],
        ['american', 'indians'],
        ['alaska', 'native'],
        ['alaska', 'natives']
    ],
    'Asian': [
        ['asian'],
        ['asians'],
    ],
    'Black or African American': [
        ['african', '-', 'american'],
        ['black'],
        ['african', 'heritage'],
        ['african', 'american'],
        ['african', '-', 'americans'],
        ['blacks'],
        ['african', 'americans']
    ],
    'Hispanic or Latino': [
        ['hispanic'],
        ['latino'],
        ['latina'],
        ['hispanics'],
        ['latinos'],
        ['latinas'],
    ],
    'Not Hispanic or Latino': [
        ['not', 'hispanic'],
        ['non', '-', 'hispanic']
    ],
    'More than one race': [
        ['mixed', 'race'],
        ['mixed', 'races'],
        ['more', 'than', 'one', 'race'],
    ],
    'Other': [
        ['other']
    ],
    'Native Hawaiian or Other Pacific Islander': [
        ['native', 'hawaiian'],
        ['native', 'hawaiians'],
        ['pacific', 'islander'],
        ['pacific', 'islanders']
    ],
    'Unknown or Not Reported': [
        ['missing'],
        ['unknown'],
        ['unreported']
    ],
    'White': [
        ['caucasian'],
        ['white'],
        ['european'],
        ['caucasians'],
        ['whites'],
        ['europeans']
    ],
}

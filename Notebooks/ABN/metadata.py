""" this file contains the meta date as to ABN AMRO churn prediction project """

# Churn on which type of product package are we talkng about?
# I like meta data for a qucik sneak peek at the data & data validation & data missingness
# It is difficult to say anything definite about the average duration that churners are active customers 
# without looking at the time since they became customers (before observation started)
# The number of customers that churned at a given month during observation.
# Do we have recurrent churn or subscription? If so they should be removed


data_type = {
    # 'MONTH_PERIOD': 'str',          # 24 Months record, values months 
    'CHURNED_IND': 'int8',            # Client churned or not, values 0 & 1
    'COMMERCIALLY_CHURNED': 'int8',   # Differ churn in   or not, values 0 & 1
    'PAYMENT_IND': 'int8',            # Client has payed, values 0 & 1
    'SAVING_IND': 'int8',             # Client has savings at bank, 0 & 1
    'INVESTMENTS_IND': 'int8',        # Client has invetment records, 0 & 1
    'LENDING_IND': 'int8',            # Client has lending records, 0 & 1
    'INSURANCE_LIFE_IND': 'int8',     # Client has life insurance records, 0 & 1
    'INSURANCE_NONLIFE_IND': 'int8',  # Client has non life insurance records, 0 & 1
    'MORTGAGE_IND': 'int8',           # Client has mortgage, 0 & 1
    'CROSS_SELL_SCORE': 'int8',       # Client score in terms of response to cross sell attempt, values 0 to 7
    'PACKAGE_IND': 'int8',            # Client subscripe to cetrain package, 0 & 1
    'CREDIT_CLASS': 'int8',           # Client assigned credit class, values 0 to 8
    'DEBIT_CLASS': 'int8',            # Client assigned debit class, values 0 to 5
    'BUSINESS_VOLUME_CLASS': 'int8',  # Client business volumne class, values 0 to 9
    'INVESTED_CAPITAL_CLASS': 'int8', # Client invested capital class, values 0 to 9
    'SAVINGS_CAPITAL_CLASS': 'int8',  # Client invested capital class, values 0 to 10
    'MIN_FEED_CLASS': 'int8',         # Client minimum feed class, 0 to 6
    'REVENUES_CLASS': 'int8',         # Client revenue class, values 0 to 19
    'PAYMENT_ACTIVITIES_CODE': 'int8',# Client paymeny activity code, values 0 to 4
    'CLIENTGROUP': 'category',        # Group to which client belongs, 48 unique groups
    'ACCOUNTMODEL': 'category',       # Client account model, acceptable categories {LP, HW, MP, HP}, [Why missingness?]
    'AGE_CLASS': 'category',          # Client age groups,  [Why missingness?], unknown
                                      # [12, 17], [18, 23], [24-29], [30-34], [35-40], [41-54], [55-64], [65-74], [75+], [Leeftijd_onbekend]
    'HOMEBANK_COLOUR': 'category',    # Client home bank colour, 'Geel', 'Groen', 'Oranje', 'Rood', missing value exist. [Why missingness?]
    'LOYALITY': 'category',           # Client loyalty colour, 'Oranje', 'Groen', 'Rood', 'Wit', [Why missingness?]
    'TARGET': 'int8',                 # Client Target, constant category 99
    'Record_Count': 'int8',           # Number of records available per client, constant 24
    'CUSTOMER_ID': 'category',        # Client ID 
}
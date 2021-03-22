import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from env import host, user, password

# Establish a connection
def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the CodeUp db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_latitude(df):
    '''
    This function takes in a datafame with latitude formatted as a float,
    converts it to a int and utilizes lambda to return the latitude values
    in a correct format.
    '''
    df.latitude = df.latitude.astype(int)
    df['latitude'] = df['latitude'].apply(lambda x: x / 10 ** (len((str(x))) - 2))
    return df

def get_longitude(df):
    '''This function takes in a datafame with longitude formatted as a float,
    converts it to a int and utilizes lambda to return the longitude values
    in the correct format.
    '''
    df.longitude = df.longitude.astype(int)
    df['longitude'] = df['longitude'].apply(lambda x: x / 10 ** (len((str(x))) - 4))
    return df

def get_county(df):
    county = []

    for row in df['fips']:
        if row == 6037:
            county.append('Los Angeles')
        elif row == 6059:
            county.append('Orange')
        elif row == 6111:
            county.append('Ventura')
        
    df['county'] = county
    return df

def calculate_tax_rate(df):
    calc_tax_rate = (df.taxamount/df.taxvaluedollarcnt)
    return df.assign(tax_rate=calc_tax_rate)

def remove_tax_value_outliers(df):
    # Calculate the interquartile range for your column

    q1, q3 = df.taxvaluedollarcnt.quantile([.25, .75])
    
    iqr = q3 - q1
    
    # Create variables holding upper and lower cutoff values using common formula. Tweak as you like.
    
    tax_upperbound = q3 + 3.5 * iqr
    
    #tax_lowerbound = q1 - 3 * iqr ==> The lowerbound is negative and since there are no negative values, 
    # there are no lowerbound outliers
    
    # Filter the column using variables and reassign to your dataframe.
    df = df[df.taxvaluedollarcnt < tax_upperbound]
    
    return df

def remove_square_feet_outliers(df):
    # Calculate the interquartile range for your column

    q1, q3 = df.calculatedfinishedsquarefeet.quantile([.25, .75])
    
    iqr = q3 - q1
    
    # Create variables holding upper and lower cutoff values using common formula. Tweak as you like.
    
    sq_upperbound = q3 + 3 * iqr
    
    sq_lowerbound = q1 - 3 * iqr
    
    # Filter the column using variables and reassign to your dataframe.
    df = df[df.calculatedfinishedsquarefeet < sq_upperbound]
    df = df[df.calculatedfinishedsquarefeet > sq_lowerbound]
    
    return df

def clean_zillow(df):
    '''
    This function reads in the zillow dataframe with 15 columns and 24950 rows
    from my acquire module and cleans it by: 

        - Setting index to parcelid
        - Replace missing values that could be calculated from other fields:
            -calculatedbathnbr
            -structuretaxdollarvaluecnt
        -Formatting latitude and longitude columns correctly
        -Droping remaining observations with missing values.

    It returns a dataframe with 14 columns and 24947 rows.
    '''
    # Set parcelid as the index
    df = df.set_index('parcelid')
    
    #Replace missing values in calculatedbathnbr with corresponding batrhoomcnt
    df.calculatedbathnbr = df.calculatedbathnbr.fillna(df.bathroomcnt)
    
    #Replace missing values in structuretaxvaluedollarcount = taxvaluedollarcnt - landtaxvaluedollarcnt
    df.structuretaxvaluedollarcnt.fillna((df.taxvaluedollarcnt - df.landtaxvaluedollarcnt), inplace=True)
    
    #Call get_latitude fucntion to clean latitude
    get_latitude(df)
    
    #Call get_longitude function to clean longitude
    get_longitude(df)

    #Call get_county function to add in County Values by FIPS
    df = get_county(df)

    #Call calculate_tax_rate function to calculate tax rates
    df = calculate_tax_rate(df)
    
    #Drop observations with missing values
    df.dropna(inplace=True)

    return df

def prepare_zillow(df):
    '''
    This function  loads in a zillow  dataframe, utilizes the
    clean_zillow function and prepares it for exploration by dropping columns that
    will not be used as features during exploration or modeling.

    It returns a dataframe with 4 columns and 24947 rows.
    '''
    df = clean_zillow(df)
    
    cols_to_drop = ['unitcnt', 'propertylandusetypeid', 'propertylandusedesc','landtaxvaluedollarcnt','taxamount', 
                'structuretaxvaluedollarcnt', 'longitude', 'latitude', 'fips', 'calculatedbathnbr', 'county']
    
    df.drop(columns=cols_to_drop, inplace=True)

    #Drop tax value outliers using IQR
    df = remove_tax_value_outliers(df) 
    
    return df


def split_stratify_continuous(df, target, bins=5):
    '''
    This function splits a data frame into train, test, validate
    and startifies by a continuous target variable.
    '''
    binned_y = pd.cut(df[target], bins=bins, labels=list(range(bins)))
    df["bins"] = binned_y
    train_validate, test = train_test_split(df, stratify=df["bins"], test_size=0.2, random_state=123)
    train, validate = train_test_split(train_validate, stratify=train_validate["bins"], test_size=0.3, random_state=123)
    train = train.drop(columns=["bins"])
    validate = validate.drop(columns=["bins"])
    test = test.drop(columns=["bins"])
    return train, test, validate

def prepare_zillow_2nd(df):
    '''
    This function  loads in a zillow  dataframe, utilizes the
    clean_zillow function and prepares it for exploration by dropping columns that
    will not be used as features during exploration or modeling.

    It returns a dataframe with 4 columns and 24947 rows.
    '''
    df = clean_zillow(df)
    
    cols_to_drop = ['unitcnt', 'propertylandusetypeid','landtaxvaluedollarcnt','taxamount', 
                'structuretaxvaluedollarcnt', 'fips', 'calculatedbathnbr']
    
    #cols_to_keep = ['propertylandusedesc', 'longitude', 'latitude', 'county']

    df.drop(columns=cols_to_drop, inplace=True)

    #Drop tax value outliers using IQR
    df = remove_tax_value_outliers(df) 
    
    return df
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
                'structuretaxvaluedollarcnt', 'longitude', 'latitude', 'fips', 'calculatedbathnbr']
    
    df.drop(columns=cols_to_drop, inplace=True)
    
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
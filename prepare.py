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

# Acquire Data
def get_zillow_data():
    '''
    This function reads in zillow data from Codeup database and returns it
    as a .csv file containing a single dataframe. 
    '''
    
    filename = "zillow.csv"
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        df = pd.read_sql('''
            SELECT prop.parcelid, 
                    prop.bathroomcnt, 
                    prop.bedroomcnt, 
                    prop.calculatedbathnbr, 
                    prop.calculatedfinishedsquarefeet, 
                    prop.fips,
                    prop.latitude,
                    prop.longitude, 
                    prop.regionidcity, 
                    prop.regionidcounty, 
                    prop.regionidzip, 
                    prop.structuretaxvaluedollarcnt, 
                    prop.taxvaluedollarcnt, 
                    prop.landtaxvaluedollarcnt, 
                    prop.taxamount, 
                    prop.taxdelinquencyflag, 
                    prop.taxdelinquencyyear,
                    prop.unitcnt
            FROM properties_2017 as prop
            JOIN predictions_2017 as pred USING(parcelid)
            JOIN propertylandusetype as proptype  ON prop.propertylandusetypeid = proptype.propertylandusetypeid
            WHERE pred.transactiondate BETWEEN '2017-05-01'
                                AND '2017-08-31'
                AND prop.unitcnt = '1';
            ''', 
            get_connection('zillow'))
        df.to_file(filename)
        return df
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
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

def bathrooms_per_squareft(df):
    calc_bath_per_sqft = (df.bathroomcnt/df.calculatedfinishedsquarefeet)
    return df.assign(bath_per_sqft=calc_bath_per_sqft)

def remove_tax_value_outliers(df):
    # Calculate the interquartile range for your column

    q1, q3 = df.taxvaluedollarcnt.quantile([.25, .75])
    
    iqr = q3 - q1
    
    # Create variables holding upper and lower cutoff values using common formula. Tweak as you like.
    
    tax_upperbound = q3 + 3 * iqr
    
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

    #Convert fips to int
    df.fips = df.fips.astype('int64')

    #Drop observations with missing values
    df.dropna(inplace=True)

    #Call get_county function to add in County Values by FIPS
    df = get_county(df)

    #Call calculate_tax_rate function to calculate tax rates
    df = calculate_tax_rate(df)

    #Call bathrooms_per_squarefeet function to calculate the rate of ba per sqft
    df = bathrooms_per_squareft(df)
    
    return df

def prepare_zillow(df):
    '''
    This function  loads in a zillow  dataframe, utilizes the
    clean_zillow function and prepares it for exploration by dropping columns that
    will not be used as features during exploration or modeling.

    It returns a dataframe with 4 columns and 24947 rows.
    '''
    df = clean_zillow(df)
    cols_to_drop = ['propertylandusetypeid', 'propertylandusedesc','landtaxvaluedollarcnt','taxamount', 
                'structuretaxvaluedollarcnt', 'longitude', 'latitude', 'fips', 'calculatedbathnbr', 'county', 'tax_rate',
                'bath_per_sqft']
    
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
    
    cols_to_drop = ['propertylandusetypeid','landtaxvaluedollarcnt','taxamount', 
                'structuretaxvaluedollarcnt', 'fips', 'calculatedbathnbr']
    
    #cols_to_keep = ['propertylandusedesc', 'longitude', 'latitude', 'county']

    df.drop(columns=cols_to_drop, inplace=True)

    #Drop tax value outliers using IQR
    df = remove_tax_value_outliers(df) 
    
    return df

def get_object_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # create a mask of columns whether they are object type or not
    mask = np.array(df.dtypes == "object")

        
    # get a list of the column names that are objects (from the mask)
    object_cols = df.iloc[:, mask].columns.tolist()
    
    return object_cols

def create_dummies(df, object_cols):
    '''
    This function takes in a dataframe and list of object column names,
    and creates dummy variables of each of those columns. 
    It then appends the dummy variables to the original dataframe. 
    It returns the original df with the appended dummy variables. 
    '''
    
    # run pd.get_dummies() to create dummy vars for the object columns. 
    # we will drop the column representing the first unique value of each variable
    # we will opt to not create na columns for each variable with missing values 
    # (all missing values have been removed.)
    dummy_df = pd.get_dummies(df[object_cols], dummy_na=False, drop_first=True)
    
    # concatenate the dataframe with dummies to our original dataframe
    # via column (axis=1)
    df = pd.concat([df, dummy_df], axis=1)

    return df

def train_validate_test(train, validate, test, target, bins=5):
    '''
    this function takes in a dataframe and splits it into 3 samples, 
    a test, which is 20% of the entire dataframe, 
    a validate, which is 24% of the entire dataframe,
    and a train, which is 56% of the entire dataframe. 
    It then splits each of the 3 samples into a dataframe with independent variables
    and a series with the dependent, or target variable. 
    The function returns 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test. 
    '''
    
    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=[target])
    y_train = train[target]
    
    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]
    
    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target])
    y_test = test[target]
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test

def get_numeric_X_cols(X_train, object_cols):
    '''
    takes in a dataframe and list of object column names
    and returns a list of all other columns names, the non-objects. 
    '''
    numeric_cols = [col for col in X_train.columns.values if col not in object_cols]
    
    return numeric_cols

def min_max_scale(X_train, X_validate, X_test, numeric_cols):
    '''
    this function takes in 3 dataframes with the same columns, 
    a list of numeric column names (because the scaler can only work with numeric columns),
    and fits a min-max scaler to the first dataframe and transforms all
    3 dataframes using that scaler. 
    it returns 3 dataframes with the same column names and scaled values. 
    '''
    # create the scaler object and fit it to X_train (i.e. identify min and max)
    # if copy = false, inplace row normalization happens and avoids a copy (if the input is already a numpy array).


    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(X_train[numeric_cols])

    #scale X_train, X_validate, X_test using the mins and maxes stored in the scaler derived from X_train. 
    # 
    X_train_scaled_array = scaler.transform(X_train[numeric_cols])
    X_validate_scaled_array = scaler.transform(X_validate[numeric_cols])
    X_test_scaled_array = scaler.transform(X_test[numeric_cols])

    # convert arrays to dataframes
    X_train_scaled = pd.DataFrame(X_train_scaled_array, 
                                  columns=numeric_cols).\
                                  set_index([X_train.index.values])

    X_validate_scaled = pd.DataFrame(X_validate_scaled_array, 
                                     columns=numeric_cols).\
                                     set_index([X_validate.index.values])

    X_test_scaled = pd.DataFrame(X_test_scaled_array, 
                                 columns=numeric_cols).\
                                 set_index([X_test.index.values])

    
    return X_train_scaled, X_validate_scaled, X_test_scaled
# Zillow Regression Project
## Building a Predictive Model for Property Values

### About the Project
**Big Idea:** Can I build a machine learning model using regression to predict a single unit property's value that performs better than the baseline model?

**How is success defined:** The primary goal of this project is to utilize and hone the tools I have acquired as a data scientist by working through the data science pipeline. The secondary goal is to develop a model utilizing statistically justifiable features that performs at least as well if not better than the baseline model for predicting a single unit property's value. 

**Project Presentation in Tableau:** Click <here>(https://public.tableau.com/profile/justin.sullivan#!/vizhome/Zillow_Regression_Project/Presentation) to view my project slide deck I created and presented using Tableau.

### What is included in this repo?

- This Readme file:
    - Project Overview and Key Takeaways
    - Data Dictionary
    - Skills and tools necessary for replication
    - Outline of process and tips for replication
    
- Necessary Modules as .py files
    -acquire.py
    -prepare.py
    -evaluate.py
    -explore.py
    -preprocess.py
    
- Report notebook with highlights from my process
    - Key takwaways from Data Acquisition
    - How I cleaned and prepared the data
    - Exploratory analysis and key questions
    - Hypotheses regarding drivers of the target and statistical testing
    - Visualizations of features and targets (univariate, bivariate and multivariate)
    - Feature selection and feature engineering
    - Development of a baseline model
    - Model development and evaluation metrics
    
- XLSL file for exporting dataframe to Tableau for visulaizations

### Data Dictionary

| Target| Description | Data Type |
|---------|-------------|-----------|
| 'taxvaluedollarrcnt' | The total tax assessed value of the parcel (THE VALUE) | float64 |

| Features | Description | Data Type |
|---------|-------------|-----------|
| 'parcelid' | Index: Unique identifier for each property  | int64 |
| 'bathroomcnt' | Indicates the number of bathrooms a property has and includes fractional bathrooms | float64 |
| 'bedroomcnt' | Indicates the number of bedrooms a property has | float64 |
| 'calculatedbathnbr' | Indicates the number of bathrooms and includes fractional bathrooms| float64 |
| 'calculatedfinishedsquarefeet' | Calculated total finished living area of the property  | float64 |
| 'fips' | Federal Information Processing Standard code -  see https://en.wikipedia.org/wiki/FIPS_county_code for more details  | int64 |
| 'latitude' |  Latitude of the middle of the parcel| float64 |
| 'longitude' |  Longitude of the middle of the parcel  | float64 |
| 'tax_rate' | Calculated tax rate for the property| float64 |
| 'bath_per_sqft' | Calculated baths per finished square feet of the living area | float64 |
'structuretaxvaluedollarcnt' | tax value of the finished living area on the property| int64 |
'landtaxvaluedollarcnt' | Tax value of the land area of the parcel | int 64 |
'taxamount' | The total property tax assessed for that assessment year | int64 |
'propertylandusetypeid' |  Type of land use the property is zoned for | int64 |
'propertylandusedesc' | Description of the allowed land uses (zoning) for that property | object |


### Project Replication Tips

#### Technical Skills
- Python
    - Pandas
    - Seaborn
    - Matplotlib
    - Numpy
    - Sklearn
        -Preprocessing
        -Feature_Selection
            -SelectKBest
            -Recursive Feature Elimination
    
- SQL

- Statistical Analysis
    - Descriptive Stats
    - Hypothesis Testing
        - Pearsons Correlation Testing
        
- Regression Modeling
    - Linear Regression
    - LASSO + LARS
    - Generalized Linear Model (TweedieRegressor)
    - Baseline Accuracy

    
#### Modules Included in Repo
- acquire.py
- prepare.py
- explore.py

### Project Process

**Trello Board**
My process and steps that I took can be viewed on my Trello board at this link:
https://trello.com/invite/b/5UAXYMrN/6df7dfe75d97cd4c5f9be7738376d19f/zillow-regression-project


### Key Findings

- **Number of Bathrooms, Number of Bedrooms, and Finished Living Area Square Feet are the top three drivers of property value.**
- **The mean baseline is $449682.262 and the baseline model performance can be evaluated by an RMSE of $348,740**
- **The GLM model performs better than the baseline model with a RMSE of $289,445 on out of sample data.**


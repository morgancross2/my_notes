import numpy as np
import pandas as pd
import os
import acquire
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def prep_iris(df):
    '''
    This function accepts the iris df and preps it.
    '''
    df = df.drop(columns=['species_id', 'measurement_id'])
    df = df.rename(columns={'species_name':'species'})
    dummies = pd.get_dummies(df.species)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns='species')
    
    return df


def prep_titanic(df):
    '''
    This function accepts the titanic df and preps it. 
    '''
    df = df.drop(columns='embarked')
    df = df.drop(columns='class')
    df = df.drop(columns=['age','deck'])
    dummy_df = pd.get_dummies(df[['sex', 'embark_town']], drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    df = df.drop(columns=['sex', 'embark_town'])
    
    return df


def prep_telco(telco):
    '''
    This function accepts the telco df and preps it.
    '''
    # fix monthly charges from string to floats
    telco.total_charges = telco.total_charges.str.replace(' ','0')
    telco.total_charges = telco.total_charges.astype('float')
    
    # drop foreign key columns
    telco = telco.drop(columns=['internet_service_type_id', 'payment_type_id', 'contract_type_id'])
    
    # split out 0 vs 1 columns
    dummies = pd.get_dummies(telco[['gender','partner', 'dependents', 'phone_service','paperless_billing', 'churn']], drop_first=True)
    telco = pd.concat([telco, dummies], axis=1)
    telco = telco.drop(columns=['gender', 'partner', 'dependents', 'phone_service','paperless_billing', 'churn'])
    
    # split out category columns
    dum = pd.get_dummies(telco.contract_type)
    telco = pd.concat([telco, dum], axis=1)
    telco = telco.drop(columns='contract_type')
    dum = pd.get_dummies(telco.internet_service_type)
    telco = pd.concat([telco, dum[['DSL', 'Fiber optic']]], axis=1)
    dum = pd.get_dummies(telco.payment_type)
    telco = pd.concat([telco, dum], axis=1)
    telco = telco.drop(columns=(['payment_type', 'internet_service_type']))
    
    # relabel final columns with 0 or 1 values
    telco.multiple_lines = telco.multiple_lines.str.replace('No phone service', '0').str.replace('Yes', '2').str.replace('No', '1').astype('int')
    telco.online_security = telco.online_security.str.replace('No internet service', '0').str.replace('Yes', '1').str.replace('No', '0').astype('int')
    telco.online_backup = telco.online_backup.str.replace('No internet service', '0').str.replace('Yes', '1').str.replace('No', '0').astype('int')
    telco.device_protection = telco.device_protection.str.replace('No internet service', '0').str.replace('Yes', '1').str.replace('No', '0').astype('int')
    telco.tech_support = telco.tech_support.str.replace('No internet service', '0').str.replace('Yes', '1').str.replace('No', '0').astype('int')
    telco.streaming_tv = telco.streaming_tv.str.replace('No internet service', '0').str.replace('Yes', '1').str.replace('No', '0').astype('int')
    telco.streaming_movies = telco.streaming_movies.str.replace('No internet service', '0').str.replace('Yes', '2').str.replace('No', '0').astype('int')
    
    return telco


def split_data(df):
    '''
    Takes in a dataframe and returns train, validate, and test subset dataframes 
    with the .2/.8 and .25/.75 splits to create a final .2/.2/.6 split between datasets
    '''
    train, test = train_test_split(df, test_size = .2, random_state=123)
    train, validate = train_test_split(train, test_size = .25, random_state=123)
    
    return train, validate, test


def impute_mode(train, validate, test, col):
    '''
    Takes in train, validate, and test as dfs, and column name (as string) and uses train 
    to identify the best value to replace nulls in embark_town
    
    Imputes the most_frequent value into all three sets and returns all three sets
    '''
    imputer = SimpleImputer(strategy='most_frequent')
    imputer = imputer.fit(train[[col]])
    train[[col]] = imputer.transform(train[[col]])
    validate[[col]] = imputer.transform(validate[[col]])
    test[[col]] = imputer.transform(test[[col]])
    
    return train, validate, test
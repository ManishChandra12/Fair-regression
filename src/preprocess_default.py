# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:03:22 2020
Name - aAnirban Saha
@author: A
"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import random
import os

def main():
    
#import csv as csv
######################################################
    deflt1 = pd.read_excel("data/raw/Default/default of credit card clients.xls" , skiprows=1)
    deflt1.to_csv(r'data/processed/Default/default_cc_f1.csv', index=False)
    # The data to load
    f = "data/processed/Default/default_cc_f1.csv"
    # Count the lines
    num_lines = sum(1 for l in open(f))
    # Sample size - in this case ~5%
    size = int(num_lines / 20)
    # The row indices to skip - make sure 0 is not included to keep the header!
    skip_idx = random.sample(range(1, num_lines), num_lines - size)
    # Read the data
    data = pd.read_csv(f, skiprows=skip_idx)
    #write to file
    data.to_csv(r'data/processed/Default/default_cc_RandomSample.csv', index=False)
    
    ######################################################
    deflt1 = pd.read_csv("data/processed/Default/default_cc_RandomSample.csv")
    ######################################################
    deflt2 = deflt1.copy()
    deflt3 = deflt1.copy()
    
    deflt1=deflt1.fillna("0")
    #Copy the colum named SEX into a new column named SEX_MALE
    deflt1['SEX_MALE']=deflt1['SEX']
    #Only keep the colum value 1 which is equal to male . So we are basically
    #making column value 2 which is female equal to 0
    deflt1.loc[(deflt1.SEX_MALE == 2),'SEX_MALE']=0
    
    
    #Copy the column named SEX into a new column named SEX_FEMALE
    deflt1['SEX_FEMALE']=deflt1['SEX']
    #making the column SEX_FEMALE value 1 equal to 0 
    deflt1.loc[(deflt1.SEX_FEMALE == 1),'SEX_FEMALE']=0
    #making the column SEX_FEMALE value 2 equal to 1
    deflt1.loc[(deflt1.SEX_FEMALE == 2),'SEX_FEMALE']=1
    deflt1['default_payment_next_month']=deflt1['default payment next month']
    
    deflt1.drop('default payment next month', axis=1, inplace=True)
    # does the dataframe have any missing value
    assert deflt1.isnull().values.any() == False
    
    default1=deflt1.fillna(0) 
    default2=deflt1.fillna(0) 
    default3=deflt1.fillna(0) 
    deflt1.drop('SEX', axis=1, inplace=True)
    deflt1.to_csv(r'data/processed/Default/default_processed.csv' , index=False)
    
    
    # Drop the column with label 'SEX'                  
    default1.drop('SEX', axis=1, inplace=True)
    #default1.to_csv(r'pre_process_group.csv' , index=False)
    #default1 = default1.iloc[1:]
    
    # gender is equal to 1 
    default2_filtered1 = default2[default2['SEX_MALE'] == 1]
    # Drop the column with label 'SEX'  
    default2_filtered1.drop('SEX_MALE', axis=1, inplace=True)
    default2_filtered1.drop('SEX', axis=1, inplace=True)
    #print (default2_filtered1.head(15)) 
    
    default2_filtered1.to_csv(r'data/processed/Default/default_Sex-Male.csv', index=False)
    
    # gender is equal to 2
    default2_filtered2 = default2[default2['SEX_FEMALE'] == 1]
    # Drop the column with label 'SEX'  
    default2_filtered2.drop('SEX_FEMALE', axis=1, inplace=True) 
    #print (default2_filtered2.head(15))
    default2_filtered2.drop('SEX', axis=1, inplace=True)
    default2_filtered2.to_csv(r'data/processed/Default/default_Sex-Female.csv', index=False)
    
    #Remove temporary file 'default_cc_f1.csv'
    os.remove("data/processed/Default/default_cc_f1.csv") 
#
#print(default1.isnull().sum())
#print(default2_filtered1.isnull().sum())
#print(default2_filtered2.isnull().sum())
if __name__ == '__main__':
	main()
 





import pandas as pd
import numpy as np
import chardet

from functools import partial

def read_CSV():
    data_path = "home-depot-product-search-relevance/"
   
    df_attributes = pd.read_csv(data_path + "attributes.csv")
    df_product_descriptions = pd.read_csv(data_path + "product_descriptions.csv")
    df_sample_submission = pd.read_csv(data_path + "sample_submission.csv")
    
    with open(data_path + "test.csv", 'rb') as f:
        enc = chardet.detect(f.read()) 
    
    df_test = pd.read_csv(data_path + "test.csv", encoding = enc['encoding'])
    df_train = pd.read_csv(data_path + "test.csv", encoding = enc['encoding'])
    
    return df_attributes, df_product_descriptions, df_sample_submission, df_test, df_train

def data_exploration(df_attributes):
    #1. What is the total number of product-query pairs in the training data?
    
    #2. What is the number of unique products in the training data?
    df_attributes['species'].nunique()
    
    #3. What are the two most occurring products in the training data and how often do they occur?
    
    #4. Give the descriptive statistics for the relevance values (mean, median, standard deviation) in the training data.
    
    #5. Show a histogram or boxplot of the distribution of relevance values in the training data.
    
    #6. What are the top-5 most occurring brand names in the product attributes?
    










def main():
    read_CSV()
    

if __name__ == "__main__":
    main()
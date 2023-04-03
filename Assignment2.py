import pandas as pd
import numpy as np
import chardet
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer


def read_CSV():
    data_path = "home-depot-product-search-relevance/"
   
    df_attributes = pd.read_csv(data_path + "attributes.csv")
    df_product_descriptions = pd.read_csv(data_path + "product_descriptions.csv")
    df_sample_submission = pd.read_csv(data_path + "sample_submission.csv")
    
    with open(data_path + "test.csv", 'rb') as f:
        enc = chardet.detect(f.read()) 
    
    df_test = pd.read_csv(data_path + "test.csv", encoding = enc['encoding'])
    df_train = pd.read_csv(data_path + "train.csv", encoding = enc['encoding'])
    
    return df_attributes, df_product_descriptions, df_sample_submission, df_test, df_train

def data_exploration(df_attributes, df_train):
    #1. What is the total number of product-query pairs in the training data?
    #id - a unique Id field which represents a (search_term, product_uid) pair
    print("#1 The total number of product-query pairs in the training data:",  len(df_train))
    print("-----------------------------------------------------------------------------------------")
    
    #2. What is the number of unique products in the training data?
    print("#2 The number of unique products in the training data:", df_train['product_title'].nunique())
    print("-----------------------------------------------------------------------------------------")
    
    #3. What are the two most occurring products in the training data and how often do they occur?
    df_count = (df_train['product_title'].value_counts().to_frame()).reset_index()
    df_count.rename(columns = {'index': 'product_title', 'product_title': 'count'}, inplace = True)
    print("#3 The two most occurring products in the training data are:")
    print(df_count.iloc[0][0], ":", df_count.iloc[0][1], " occurrences")
    print(df_count.iloc[1][0], ":", df_count.iloc[1][1], " occurrences")
    print(df_count.iloc[2][0], ":", df_count.iloc[2][1], " occurrences")
    print(df_count.iloc[3][0], ":", df_count.iloc[3][1], " occurrences")
    print("-----------------------------------------------------------------------------------------")
    
    
    #4. Give the descriptive statistics for the relevance values (mean, median, standard deviation) in the training data.
    print("The descriptive statistics for the relevance values (mean, median, standard deviation) in the training data:")
    print("Mean:", df_train['relevance'].mean())
    print("Median:", df_train['relevance'].median())
    print("Standard deviation:", df_train['relevance'].std())
    print("-----------------------------------------------------------------------------------------")

    #5. Show a histogram or boxplot of the distribution of relevance values in the training data.
    df_train.hist(column = 'relevance', color = 'blue')
    plt.xlabel('Relevance score')
    plt.ylabel('Frequency')
    plt.title('Histogram of the distribution of relevance values in the training data')
    plt.show()
    
    # Maybe add a density plot
    df_train.boxplot(column = 'relevance', color = 'blue')
    plt.title('Boxplot of the distribution of relevance values in the training data')
    plt.show()
    
    #6. What are the top-5 most occurring brand names in the product attributes?
    df_names_count = (df_attributes['name'].value_counts().to_frame()).reset_index()
    df_names_count.rename(columns = {'index': 'name', 'name': 'count'}, inplace = True)
    print("#6 The top-5 most occurring brand names in the product attributes:")
    print(df_names_count.iloc[0][0], ":", df_names_count.iloc[0][1], " occurrences")
    print(df_names_count.iloc[1][0], ":", df_names_count.iloc[1][1], " occurrences")
    print(df_names_count.iloc[2][0], ":", df_names_count.iloc[2][1], " occurrences")
    print(df_names_count.iloc[3][0], ":", df_names_count.iloc[3][1], " occurrences")
    print(df_names_count.iloc[4][0], ":", df_names_count.iloc[4][1], " occurrences")
    print("-----------------------------------------------------------------------------------------")


##################### The code by Yao-Jen Chang #####################

# Stemming maps different forms of the same word to a common “stem” - for example, 
# the English stemmer maps connection, connections, connective, connected, and connecting to connect. 
# So a searching for connected would also find documents which also have the other forms.
stemmer = SnowballStemmer('english')
# For a provided string, returns a new string with the words replaced by their roots in English, e.g.
# programmer -> program 
def str_stemmer(s):
    	return " ".join([stemmer.stem(word) for word in s.lower().split()])
# Calculate the number of common words in str1 and str2
def str_common_word(str1, str2):
	return sum(int(str2.find(word)>=0) for word in str1.split())

def original_script():
    # Reading the data
    df_train = pd.read_csv('home-depot-product-search-relevance/train.csv', encoding="ISO-8859-1")
    df_test = pd.read_csv('home-depot-product-search-relevance/test.csv', encoding="ISO-8859-1")
    # df_attr = pd.read_csv('../input/attributes.csv')
    df_pro_desc = pd.read_csv('home-depot-product-search-relevance/product_descriptions.csv')

    num_train = df_train.shape[0]

    # Data preprocessing
    # Concatenating train.csv, test.csv and product_descriptions.csv files
    df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
    df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
    
    # Stemming the search_term values
    df_all['search_term'] = df_all['search_term'].map(lambda x:str_stemmer(x))
    # Stemming the product_title values
    df_all['product_title'] = df_all['product_title'].map(lambda x:str_stemmer(x))
    # Stemming the product_description values
    df_all['product_description'] = df_all['product_description'].map(lambda x:str_stemmer(x))
    # Adding a new column len_of_query, which contains the number of words in search_term
    df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)
    # Adding a new column product_info which contrains search_term + product_title + product_description
    df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title']+"\t"+df_all['product_description']
    # Adding a new column word_in_title which contains the number of matching (same) words in search_term and product_title
    df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
    # Adding a new column word_in_description which contains the number of matching (same) words in search_term and product_description
    df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))
    # Droping the following columns: search_term, product_title, product_description and product_info
    df_all = df_all.drop(['search_term','product_title','product_description','product_info'],axis=1)
    # df_all contains the following columns: 
    # id - unique Id field which represents a (search_term, product_uid) pair
    # product_uid - an id for the products
    # relevance - the average of the relevance ratings for a given id. Note, that test instances do NOT contain those values
    # len_of_query - the number of words in search_term
    # word_in_title - the number of matching (same) words in search_term and product_title
    # word_in_description - the number of matching (same) words in search_term and product_description

    # df_train contains the instances with the relevance values (first num_train instances)
    df_train = df_all.iloc[:num_train]
    # df_test contains the test instances
    df_test = df_all.iloc[num_train:]
    # id_test contrains a unique Id field which represents a (search_term, product_uid) pair
    id_test = df_test['id']

    # y_train or labels are the relevance scores
    y_train = df_train['relevance'].values
    # X_train and X_test contain product_uid, len_of_query, word_in_title and word_in_description
    X_train = df_train.drop(['id','relevance'],axis=1).values
    X_test = df_test.drop(['id','relevance'],axis=1).values

    # A random forest is a meta estimator that fits a number of classifying decision trees on various sub-samples 
    # of the dataset and uses averaging to improve the predictive accuracy and control over-fitting
    # n_estimators — the number of decision trees we will be running in the model
    # max_depth — the maximum possible depth of each tree
    rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
    # A Bagging regressor is an ensemble meta-estimator that fits base regressors each on random subsets of the original dataset 
    # and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction. 
    # Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator (e.g., a decision tree),
    # by introducing randomization into its construction procedure and then making an ensemble out of it.
    # max_samples - the number of samples to draw from X to train each base estimator
    clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)
    # Build a Bagging ensemble of estimators from the training set (X_train, y_train).
    clf.fit(X_train, y_train)
    # Predict regression target for X_test.
    y_pred = clf.predict(X_test)

    #Safe the id's and relevance scores into a CSV file.
    pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission.csv',index=False)

##################### The code by Yao-Jen Chang #####################




def main():
    df_attributes,_,_,_,df_train = read_CSV()
    data_exploration(df_attributes, df_train)
    #original_script()
    

if __name__ == "__main__":
    main()
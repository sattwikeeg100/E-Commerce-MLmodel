import pandas as pd
import scipy.sparse
import difflib

# Load ML/Rc Pkgs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel


# Load Our Dataset
def load_data(dataset):
	df = pd.read_csv(dataset)
	return df

def data_preprocess(df):
    # dropping the unnecessary columns
    df.drop(['course_image','url','is_paid','price','num_subscribers','num_reviews','num_lectures','course_id','published_timestamp','profit','published_date','published_time','year','month','day'], axis=1,inplace=True)
    # Filling NaNs with empty string
    df['cleaned'] = df['subject'].fillna('')
    return df

def recommender(input):
    # loading the data
    df_org = load_data("data/course_data.csv")
    df = df_org.copy()  # copying the dataset to perform preprocessing opertaions on it
    
    # preprocessing the data
    df = data_preprocess(df)

    # generating the simlist
    namelist=df['course_title'].tolist()
    simlist=difflib.get_close_matches(input, namelist)
    title= simlist[0]

    ## vectroize our text
    tfv = TfidfVectorizer(min_df=3,  max_features=None, 
                strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
                ngram_range=(1, 3),
                stop_words = 'english')
    tfv_matrix = tfv.fit_transform(df['cleaned'])

    # Compute the sigmoid kernel
    sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
    
    # Reverse mapping of indices and titles
    indices = pd.Series(df.index, index=df['course_title']).drop_duplicates()
    
    # Get the index corresponding to given course
    idx = indices[title]
    
    # Get the pairwsie similarity scores with given course with every available course in the data set
    sig_scores = list(enumerate(sig[idx]))
    
    # Sort the recommended courses
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
    
    # Scores of the 10 most similar courses
    sig_scores = sig_scores[1:11]
    
    # get courses indices for top 10 recommended courses
    course_indices = [i[0] for i in sig_scores]
    
    # Top 10 most similar courses
    return df_org.iloc[course_indices]

def recommend_course(course):
    try:
        findf=recommender(course)
        findf=findf.reset_index(drop=True)
        findf.drop(['num_subscribers','num_reviews','num_lectures','course_id','published_timestamp','profit','published_time','year','month','day'], axis=1,inplace=True)
    except:
        findf = pd.DataFrame()
        
    return findf

# Bitcoin Profits for Beginners
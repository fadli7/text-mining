#%%
import pandas as pd
import numpy
from sklearn.feature_extraction.text import CountVectorizer
import re
from sklearn.feature_extraction.text import TfidfTransformer

def get_stop_words(stop_file_path):
    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)

        return frozenset(stop_set)

def pre_process(text):
    text = text.lower()
    text = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ", text)
    text = re.sub("(\\d|\\W)+", " ", text)

    return text

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return  sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_name, sorted_items, topn=10):
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        score_vals.append(round(score, 3))
        feature_vals.append(feature_name[idx])
    
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]
    
    return results

def main():
    """ Data Training """
    df_idf = pd.read_json("data/stackoverflow-data-idf.json", lines=True)

    print(df_idf['body'])

    df_idf['text'] = df_idf['title'] + df_idf['body']
    df_idf['text'] = df_idf['text'].apply(lambda x: pre_process(x)) # tokenizing

    stopwords = get_stop_words("resources/stopwords.txt")
    docs = df_idf['text'].tolist()
    cv = CountVectorizer(stop_words=stopwords)
    word_count_vector = cv.fit_transform(docs) #filtering
    
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)


    """ Data Test """
    # MARK -> Input User
    sentences = input("\n\nEnter Your Sentences : ")
    feature_name = cv.get_feature_names()

    tf_idf_vector = tfidf_transformer.transform(cv.transform([sentences]))
    sorted_items = sort_coo(tf_idf_vector.tocoo())
    keywords = extract_topn_from_vector(feature_name, sorted_items, 10)

    
    
    print("\n======= Doc =======")
    print(sentences)

    print("\n======= Keywords =======")
    for k in keywords:
        print(k, keywords[k])

if __name__ == "__main__":
    main()
#%%
import pandas as pd
import numpy
from sklearn.feature_extraction.text import CountVectorizer
import re
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

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

def sortFirst(val):
    return val[0]

def main():
    """ Data Training """
    df_idf = pd.read_json("data/stackoverflow-data-idf.json", lines=True)

    df_idf['text'] = df_idf['title'] + df_idf['body']
    df_idf['text'] = df_idf['text'].apply(lambda x: pre_process(x)) # tokenizing

    stopwords = get_stop_words("resources/stopwords.txt")
    docs = df_idf['text'].tolist()
    cv = TfidfVectorizer(stop_words=stopwords)
    data_train = cv.fit_transform(docs) #filtering
    
    # tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    # tfidf_transformer.fit(word_count_vector)


    """ Data Test """
    # MARK -> Input User
    sentences = input("\n\nEnter Your Sentences : ")

    data_test = cv.transform([sentences])

    similiarities = linear_kernel(data_train, data_test)

    data_similiarities = []
    for i in range(0, len(similiarities)):
        if similiarities[i][0] != 0:
            data_similiarities.append((similiarities[i][0], i))
    data_similiarities.sort(key=sortFirst, reverse=True)

    print("\nResult For Search Engine")
    for i in range(0, 3):
        print("\n===", i+1, "=== data in line", data_similiarities[i][1])
        print(docs[data_similiarities[i][1]])

if __name__ == "__main__":
    main()
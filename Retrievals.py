# This class contains the retrieval methods to be run using the
# cross- and bi-encoders.
import re
from bs4 import BeautifulSoup

stop_words = set([
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
        'you', "you're", "you've", "you'll", "you'd", 'your', 'yours',
        'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
        'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its',
        'itself', 'they', 'them', 'their', 'theirs', 'themselves',
        'what', 'which', 'who', 'whom', 'this', 'that', "that'll",
        'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
        'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
        'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
        'with', 'about', 'against', 'between', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'to', 'from',
        'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
        'again', 'further', 'then', 'once', 'here', 'there', 'when',
        'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
        'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
        't', 'can', 'will', 'just', 'don', "don't", 'should',
        "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
        'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
        "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
        "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn',
        "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan',
        "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
        "weren't", 'won', "won't", 'wouldn', "wouldn't", '-', '.'
    ])

def clean_text(text):
    """Clean the input text by removing HTML, punctuation, and stop words."""
    text = remove_html(text)
    text = remove_punctuation(text).lower()
    text = remove_stop_words(text)
    return text

def remove_html(text):
    """Remove HTML tags from the text."""
    soup = BeautifulSoup('html.parser')
    return soup.get_text(separator=' ')

def remove_punctuation (text):
    """Remove punctuation from the text."""
    return re.sub(r'[^\w\s]', '', text)

def remove_stop_words(text):
    """Remove stop words from the text."""
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def bi_retrieve(model, queries, docs):
    result = ""
    return result

'''
The following creates the cross encoder retrieval method
'''
def cross_retrieval(model, queries, docs, result_file):
    # input: 1 input sequence query[SEP]doc
    # will be
    # set queries to topics file
    # set docs to answers file
    # cross encoder only takes 1 input sequence and will output a value between [0,1] indicating the similarity of the input sentence pair
    # combine the query from result file to the top 100 docs returned
    # rerank
    result = ""
    return result
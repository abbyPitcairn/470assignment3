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


"""
Rerank query document pairs using a cross-encoder based on a result file from the bi-encoder.

Parameters:
    model: The cross-encoder model for reranking.
    queries: A dictionary mapping query IDs to their corresponding queries.
    docs: A dictionary mapping document IDs to their corresponding text.
    result_file: The path to the file containing the bi-encoder retrieval results.
    system_name: The name of the system to include in the output.
    top_k: The number of top results to return.
        
Returns:
   Results are written to a file.
"""
def cross_retrieval(model, queries, docs, result_file, system_name="my_cross_encoder_model", top_k=100):
    reranked_results = []

    # read from bi-encoder output file
    with open(result_file, 'r') as file:
        lines = file.readlines()

    input_pairs = [] # this will hold input pairs for the cross-encoder

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 4: # each line in the result file should have at least 4 parts (query id, q0, doc id, and rank)
            continue # skipping if it doesn't = something is wrong with the way the file was written

        topic_id = parts[0]
        doc_id = parts[2]

        query_text = queries.get(topic_id, None)
        doc_text = docs.get(doc_id, None)

        if query_text and doc_text:
            input_pairs.append((topic_id, doc_id, query_text, doc_text))

    if input_pairs:
        scores = model.predict([pair[2:] for pair in input_pairs]) # getting sim scores
        reranked_results = [(input_pairs[i][0], input_pairs[i][1], scores[i] for i in range(len(input_pairs)))] # combine scores w their query doc pair

    reranked_results.sort(key=lambda x: x[2], reverse=True) # ranking in descending order

    with open("result_ce_file", "w") as output_file: # have to rename file after!!!!
        for rank, (query_id, doc_id, score) in enumerate(reranked_results[:top_k], start=1): # technically top_k isn't needed bc the bi-encoder output file should only have 100 top results, doing just in case
            output_file.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {system_name}\n")
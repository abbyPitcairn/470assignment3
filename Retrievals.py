# This class contains the retrieval methods to be run using the
# cross- and bi-encoders.
import re
from bs4 import BeautifulSoup
import torch
import json

# Process with GPU instead of CPU; mostly for Abby's laptop.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Specify how many document search results to return
total_return_documents = 100

stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
              'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
              'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
              'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were',
              'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
              'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
              'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
              'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
              'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
              'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
              'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain',
              'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
              "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
              "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
              'wouldn', "wouldn't", '-', '.'}

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


# Behrooz said to improve speed, embed all documents and save them as a
# dictionary of "doc_id: vector" so we don't embed them repeatedly.
# Implemented below:
# Embed the documents in a collection and return a dictionary
# Keys: doc_ids
# Values: embedded value
def embed(docs, model):
    embedded_docs = {}
    for doc in docs:
        doc_id = doc['Id']
        doc_text = doc['Text']
        embedded_doc_text = model.encode(doc_text)  # Embed current document
        embedded_docs[doc_id] = embedded_doc_text
    return embedded_docs

# Method for retrieval with bi-encoders.
# Model: the model to use for embedding and calculating similarities
# Queries: the topic file
# Docs: the answers file
def bi_retrieve(model, topic_filepath, docs_filepath):
    print("Starting bi-retrieval method...")
    # Initialize dictionary to store results
    result = {}
    queries = json.load(open(topic_filepath))
    docs = json.load(open(docs_filepath))
    embedded_docs = embed(docs, model)

    # Cycle through queries
    for q in queries:
        q_id = q['Id']
        q_text = q['Title'] + " " + q['Body']
        q = model.encode(q_text)  # Embed current query

        # Search current query over documents
        for doc in embedded_docs:
            doc_id = doc['Id']
            # Calculate similarity for query and document
            simqd = model.similarity(q,doc)
            result[embedded_docs[doc]] = simqd
            result[q_id][doc_id] = simqd

    return {k: v for k, v in sorted(result.items(), key=lambda item: item[1], reverse=True)}


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


# Write the input results to an output file with columns for:
# query_id, 0, doc_id, rank, result for doc_id, name of run
def save_to_result_file(results, output_file):
    with open(output_file, 'w') as f:
        for query_id in results:
            dic_result = results[query_id]
            rank = 1  # Initialize rank to 1
            for doc_id in dic_result:
                f.write(f"{query_id} 0 {doc_id} {rank} {dic_result[doc_id]} Run1\n")
                rank += 1  # Increment the rank for each document returned
                if rank > total_return_documents:  # Only return the top 100 results from search
                    break
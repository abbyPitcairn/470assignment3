# Fine-tuning Bi-encoder
# Models: https://sbert.net/docs/sentence_transformer/pretrained_models.html
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses, evaluation
from torch.utils.data import DataLoader
from itertools import islice
import json
import torch
import math
import string
import csv
import random
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["WANDB_DISABLED"] = "true"

def read_qrel_file(file_path):
    # Reading the qrel file
    dic_topic_id_answer_id_relevance = {}
    with open(file_path) as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for row in rd:
            topic_id = row[0]
            answer_id = int(row[2])
            relevance_score = int(row[3])
            if topic_id in dic_topic_id_answer_id_relevance:
                dic_topic_id_answer_id_relevance[topic_id][answer_id] = relevance_score
            else:
                dic_topic_id_answer_id_relevance[topic_id] = {answer_id: relevance_score}
    return dic_topic_id_answer_id_relevance


def load_topic_file(topic_filepath):
    # a method used to read the topic file for this year of the lab;
    # to be passed to BERT/PyTerrier methods
    queries = json.load(open(topic_filepath))
    result = {}
    for item in queries:
      # You may do additional preprocessing here
      # returning results as dictionary of topic id: [title, body, tag]
      title = item['Title'].translate(str.maketrans('', '', string.punctuation))
      body = item['Body'].translate(str.maketrans('', '', string.punctuation))
      tags = item['Tags']
      result[item['Id']] = [title, body, tags]
    return result


def read_collection(answer_filepath):
  # Reading collection to a dictionary
  lst = json.load(open(answer_filepath))
  result = {}
  for doc in lst:
    result[int(doc['Id'])] = doc['Text']
  return result


# Uses the posts file, topic file(s) and qrel file(s) to build our training and evaluation sets.
def process_data(queries, train_dic_qrel, val_dic_qrel, collection_dic):
    train_samples = []
    evaluator_samples_1 = []
    evaluator_samples_2 = []
    evaluator_samples_score = []

    # Build Training set
    for topic_id in train_dic_qrel:
        question = queries[topic_id]
        dic_answer_id = train_dic_qrel.get(topic_id, {})

        for answer_id in dic_answer_id:
            score = dic_answer_id[answer_id]
            answer = collection_dic[answer_id]
            if score > 1:
                train_samples.append(InputExample(texts=[question, answer], label=1.0))
            else:
                train_samples.append(InputExample(texts=[question, answer], label=0.0))
    for topic_id in val_dic_qrel:
        question = queries[topic_id]
        dic_answer_id = val_dic_qrel.get(topic_id, {})

        for answer_id in dic_answer_id:
            score = dic_answer_id[answer_id]
            answer = collection_dic[answer_id]
            if score > 1:
                label = 1.0
            elif score == 1:
                label = 0.5
            else:
                label = 0.0
            evaluator_samples_1.append(question)
            evaluator_samples_2.append(answer)
            evaluator_samples_score.append(label)

    return train_samples, evaluator_samples_1, evaluator_samples_2, evaluator_samples_score



def shuffle_dict(d):
    keys = list(d.keys())
    random.shuffle(keys)
    return {key: d[key] for key in keys}


def split_train_validation(qrels, ratio=0.9):
    # Using items() + len() + list slicing
    # Split dictionary by half
    n = len(qrels)
    n_split = int(n * ratio)
    qrels = shuffle_dict(qrels)
    train = dict(islice(qrels.items(), n_split))
    validation = dict(islice(qrels.items(), n_split, None))

    return train, validation


def train(model):

    ## reading queries and collection
    dic_topics = load_topic_file("topics_1.json")
    queries = {}
    for query_id in dic_topics:
        queries[query_id] = "[TITLE]" + dic_topics[query_id][0] + "[BODY]" + dic_topics[query_id][1]
    qrel = read_qrel_file("qrel_1.tsv")
    collection_dic = read_collection('Answers.json')
    train_dic_qrel, val_dic_qrel = split_train_validation(qrel)

    # print(train_dic_qrel)
    # print(val_dic_qrel)

    num_epochs = 100
    batch_size = 16

    # Rename this when training the model and keep track of results
    MODEL = "SAVED_MODEL_NAME"

    # Creating train and val dataset
    train_samples, evaluator_samples_1, evaluator_samples_2, evaluator_samples_score = process_data(queries, train_dic_qrel, val_dic_qrel, collection_dic)

    train_dataset = SentencesDataset(train_samples, model=model)
    train_dataloader = DataLoader(train_dataset, shuffle = True, batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)

    evaluator = evaluation.EmbeddingSimilarityEvaluator(evaluator_samples_1, evaluator_samples_2, evaluator_samples_score, write_csv="evaluation-epoch.csv")
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up

    # add evaluator to the model fit function
    model.fit(
        train_objectives =[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        use_amp=True,
        save_best_model=True,
        show_progress_bar=True,
        output_path=MODEL
    )

model = SentenceTransformer('all-MiniLM-L6-v2')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
train(model)
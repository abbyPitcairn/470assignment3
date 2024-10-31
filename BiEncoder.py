# Fine-tuning Bi-encoder
# Models: https://sbert.net/docs/sentence_transformer/pretrained_models.html
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses, evaluation
from torch.utils.data import DataLoader
from itertools import islice
import Retrievals
import json
import torch
import math
import string
import csv
import random
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["WANDB_DISABLED"] = "true"

# returns a dict mapping topic IDs to a nested dict of and IDs and their score
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


# returns dict mapping IDs to their title, body, and tags
def load_topic_file(topic_filepath):
    # a method used to read the topic file for this year of the lab;
    # to be passed to BERT/PyTerrier methods
    queries = json.load(open(topic_filepath))
    result = {}
    for item in queries:
      # Using my clean_text method that can be found in Retrievals.py
      title = Retrievals.clean_text(item["Title"])
      body = Retrievals.clean_text(item["Body"])
      tags = Retrievals.clean_text(item["Tags"])
      # returning results as dictionary of topic id: [title, body, tag]
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


#changed to 80% for training 10% for validation and 10% for testing
def split_data(qrels, train_ratio=0.8, val_ratio=0.1):
    # making sure test set is created as well
    n = len(qrels)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    # shuffle
    qrels = shuffle_dict(qrels)

    # the split
    train = dict(islice(qrels.items(), n_train))
    validation = dict(islice(qrels.items(), n_train, n_train + n_val))
    test = dict(islice(qrels.items(), n_train + n_val, None))

    return train, validation, test


def train(model):
    # reading queries and collection
    dic_topics = load_topic_file("topics_1.json")
    queries = {}
    for query_id in dic_topics:
        queries[query_id] = "[TITLE]" + dic_topics[query_id][0] + "[BODY]" + dic_topics[query_id][1]
    qrel = read_qrel_file("qrel_1.tsv")
    collection_dic = read_collection('Answers.json')
    train_dic_qrel, val_dic_qrel, test_dic_qrel = split_data(
        qrel)  # created the test set, but it is never called in this file

    with open('train_qrel.json',
              'w') as f:  # will use these in CrossEncoder, so the models are trained and validated on the same set of topics
        json.dump(train_dic_qrel, f)

    with open('val_qrel.json',
              'w') as f:  # will use these in CrossEncoder, so the models are trained and validated on the same set of topics
        json.dump(val_dic_qrel, f)

    with open('test_qrel.json',
              'w') as f:  # saving this specific test set to use in Main.py, so each result file will be produced from the same test set topics
        json.dump(test_dic_qrel, f)

    num_epochs = 50 # I overheard some people in class say this helped speed up their searches !
    batch_size = 16

    model_save_path ="./ft_bi_2024" # we don't need MODEL, that's just a string. This is similar to the process in CrossEncoder where we are saving the fine-tuned model to a folder in the current directory

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
        output_path=model_save_path,
        save_best_model=True,
        show_progress_bar=True
    )
    model.save(model_save_path)

# initializing model
# model = SentenceTransformer('all-MiniLM-L6-v2')
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
# model.to(device)
# train(model)

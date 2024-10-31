# Fine-tuning Cross-encoder
import csv
import json
import math
import torch
from sentence_transformers import CrossEncoder, losses
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from torch.utils.data import DataLoader
import BiEncoder

# Change processing to GPU instead of CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def __init__(self, model_name):
    self.model_name = model_name
    self.model = CrossEncoder(model_name)


def finetune(model_name, train_samples, valid_samples):
    # Learn how to use GPU with this!
    model = CrossEncoder(model_name)

    print("Cross encoder initialized.")

    # Adding special tokens
    tokens = ["[TITLE]", "[BODY]"]
    model.tokenizer.add_tokens(tokens, special_tokens=True)
    model.model.resize_token_embeddings(len(model.tokenizer), mean_resizing=False)
    # model.to(device)
    print("Tokenizer initialized")

    # this sets up the training
    num_epochs = 100
    model_save_path = "./ft_cr_2024"  # remember this for fine-tuning!!!
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=4)
    print("Dataloader loading training")
    # During training, we use CESoftmaxAccuracyEvaluator to measure the accuracy on the dev set.
    evaluator = CERerankingEvaluator(valid_samples, name='train-eval')
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    model.fit(train_dataloader=train_dataloader,
              evaluator=evaluator,
              epochs=num_epochs,
              warmup_steps=warmup_steps,
              output_path=model_save_path,
              save_best_model=True)

    model.save(model_save_path)


def read_qrel_file(qrel_filepath):
    # a method used to read the topic file
    result = {}
    with open(qrel_filepath, "r") as f:
        reader = csv.reader(f, delimiter='\t', lineterminator='\n')
        for line in reader:
            query_id = line[0]
            doc_id = line[2]
            score = int(line[3])
            if query_id in result:
                result[query_id][doc_id] = score
            else:
                result[query_id] = {doc_id: score}
    # dictionary of key:query_id value: dictionary of key:doc id value: score
    return result


def read_collection(answer_filepath):
  # Reading collection to a dictionary
  lst = json.load(open(answer_filepath))
  result = {}
  for doc in lst:
    result[doc['Id']] = doc['Text']
  return result


## reading queries and collection
dic_topics = BiEncoder.load_topic_file("topics_1.json")
queries = {}
for query_id in dic_topics:
    queries[query_id] = "[TITLE]" + dic_topics[query_id][0] + "[BODY]" + dic_topics[query_id][1]
qrel = read_qrel_file("qrel_1.tsv")
collection_dic = read_collection('Answers.json')

## Preparing pairs of training instances
num_topics = len(queries.keys())
number_training_samples = int(num_topics*0.8) # changed to use 80% for training


## Preparing the content
counter = 1
train_samples = []
valid_samples = {}
for qid in qrel:
    # key: doc id, value: relevance score
    dic_doc_id_relevance = qrel[qid]
    # query text
    topic_text = queries[qid]

    if counter < number_training_samples:
        for doc_id in dic_doc_id_relevance:
            label = dic_doc_id_relevance[doc_id]
            content = collection_dic[doc_id]
            if label >= 1:
                label = 1
            train_samples.append(InputExample(texts=[topic_text, content], label=label))
    else:
        for doc_id in dic_doc_id_relevance:
            label = dic_doc_id_relevance[doc_id]
            if qid not in valid_samples:
                valid_samples[qid] = {'query': topic_text, 'positive': set(), 'negative': set()}
            if label == 0:
                label = 'negative'
            else:
                label = 'positive'
            content = collection_dic[doc_id]
            valid_samples[qid][label].add(content)
    counter += 1

print("Training and validation set prepared")

# selecting cross-encoder AKA initializing the model
# using this model because it works well for finding semantic textual similarity
model_name = "cross-encoder/stsb-distilroberta-base"
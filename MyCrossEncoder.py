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
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def finetune(model, train_samples, valid_samples):
    # model = CrossEncoder(model_name)
    #
    # print("Cross encoder initialized.")

    # Adding special tokens
    print("Starting fine-tuning process...")
    tokens = ["[TITLE]", "[BODY]"]
    model.tokenizer.add_tokens(tokens, special_tokens=True)
    model.model.resize_token_embeddings(len(model.tokenizer))
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

    print("Fine-tuning completed. Saving model...")
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}.")


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

# load the training and validation sets created in BiEncoder
print("Loading training QREL data...")
train_qrel = json.load(open('train_qrel.json'))
print(f"Loaded {len(train_qrel)} training queries.")

print("Loading validation QREL data...")
val_qrel = json.load(open('val_qrel.json'))
print(f"Loaded {len(val_qrel)} validation queries.")

print("Loading collection data...")
collection_dic = read_collection('Answers.json')
print(f"Loaded {len(collection_dic)} documents.")

# prepare the queries
print("Preparing queries...")
dic_topics = BiEncoder.load_topic_file("topics_1.json")
queries = {}
for query_id in dic_topics:
    queries[query_id] = "[TITLE]" + dic_topics[query_id][0] + "[BODY]" + dic_topics[query_id][1]
print(f"Prepared {len(queries)} queries.")

def prepare_samples(train_qrel, val_qrel, queries, collection_dic):
    print("Preparing training samples...")
    train_samples = []
    for qid, doc_id_relevance in train_qrel.items():
        topic_text = queries[qid]
        for doc_id, score in doc_id_relevance.items():
            content = collection_dic[doc_id]
            label = 1 if score >= 1 else 0
            train_samples.append(InputExample(texts=[topic_text, content], label=label))
    print(f"Created {len(train_samples)} training samples.")

    print("Preparing validation samples...")
    valid_samples = {}
    for qid, doc_id_relevance in val_qrel.items():
        topic_text = queries[qid]
        if qid not in valid_samples:
            valid_samples[qid] = {'query': topic_text, 'positive': set(), 'negative': set()}
        for doc_id, score in doc_id_relevance.items():
            content = collection_dic[doc_id]
            label = 'positive' if score >= 1 else 'negative'
            valid_samples[qid][label].add(content)
    print(f"Created validation samples for {len(valid_samples)} queries.")
    return train_samples, valid_samples


print("Training and validation set prepared")

# selecting cross-encoder AKA initializing the model
# using this model because it works well for finding semantic textual similarity
# model_name = "cross-encoder/stsb-distilroberta-base"

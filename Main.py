# Run the retrieval methods to get the results files.
# Version 31.10.2024

import sys
import time

from ir_datasets.datasets.wikiclir import collection

import Retrievals
import MyCrossEncoder
from sentence_transformers import SentenceTransformer, CrossEncoder
import json
import BiEncoder
from MyCrossEncoder import collection_dic

# we need both topics files to return all results
def main(answers, topics_1, topics_2):
    print("Starting main...")

    # Load each pretrained model
    pretrained_bi_model = SentenceTransformer('multi-qa-distilbert-cos-v1')
    pretrained_cross_model = CrossEncoder('stsb-distilroberta-base')

    train_qrel = json.load(open('train_qrel.json'))
    val_qrel = json.load(open('val_qrel.json'))
    collection_dic = BiEncoder.read_collection(answers)

    dic_topics_1 = BiEncoder.load_topic_file(topics_1)
    queries_1 = {qid: "[TITLE]" + text[0] + "[BODY]" + text[1] for qid, text in dic_topics_1.items()}

    # train_samples, valid_samples = MyCrossEncoder.prepare_samples(train_qrel, val_qrel, queries_1, collection_dic)

    # New objects for referencing the fine-tuned models
    ft_bi_encoder_model = BiEncoder.train(SentenceTransformer('multi-qa-distilbert-cos-v1'))
    ft_cross_encoder_model = MyCrossEncoder.finetune(CrossEncoder('stsb-distilroberta-base'), train_qrel, val_qrel)

    start_time = time.time()
    queries_2 = {}
    if topics_2:
        dic_topics_2 = BiEncoder.load_topic_file(topics_2)
        queries_2 = {qid: "[TITLE]" + text[0] + "[BODY]" + text[1] for qid, text in dic_topics_2.items()}

    with open('test_qrel.json', 'r') as f:
        test_dic_qrel = json.load(f)

    # Call retrieval methods and output result files
    # 1. No fine-tuning on test set: bi-encoder
    result_bi_1 = Retrievals.bi_retrieve(pretrained_bi_model, test_dic_qrel, collection_dic)
    with open("result_bi_1.tsv", "w") as file:
        file.write(result_bi_1)

    # 2. No fine-tuning on topics_2 file: bi-encoder
    if queries_2:
        result_bi_2 = Retrievals.bi_retrieve(pretrained_bi_model, queries_2, collection_dic)
        with open("result_bi_2.tsv", "w") as file:
            file.write(result_bi_2)

    # 3. Fine-tuning on test set: bi-encoder
    result_bi_ft_1 = Retrievals.bi_retrieve(ft_bi_encoder_model, test_dic_qrel, collection_dic)
    with open("result_bi_ft_1.tsv", "w") as file:
        file.write(result_bi_ft_1)

    # 4. Fine-tuning on topics_2 file: bi-encoder
    if queries_2:
        result_bi_ft_2 = Retrievals.bi_retrieve(ft_bi_encoder_model, queries_2, collection_dic)
        with open("result_bi_ft_2.tsv", "w") as file:
            file.write(result_bi_ft_2)

    # 5. No fine-tuning on test set: cross-encoder
    result_ce_1 = Retrievals.cross_retrieval(pretrained_cross_model, test_dic_qrel, collection_dic, 'result_bi_1.tsv')
    with open("result_ce_1.tsv", "w") as file:
        file.write(result_ce_1)

    # 6. No fine-tuning on topics_2 file: cross-encoder
    if queries_2:
        result_ce_2 = Retrievals.cross_retrieval(pretrained_cross_model, queries_2, collection_dic, 'result_bi_2.tsv')
        with open("result_ce_2.tsv", "w") as file:
            file.write(result_ce_2)

    # 7. Fine-tuning on test set: cross-encoder
    result_ce_ft_1 = Retrievals.cross_retrieval(ft_cross_encoder_model, test_dic_qrel, collection_dic, 'result_bi_ft_1.tsv')
    with open("result_ce_ft_1.tsv", "w") as file:
        file.write(result_ce_ft_1)

    # 8. Fine-tuning on topics_2 file: cross-encoder
    result_ce_ft_2 = Retrievals.cross_retrieval(ft_cross_encoder_model, queries_2, collection_dic, 'result_bi_ft_2.tsv')
    with open("result_ce_ft_2.tsv", "w") as file:
        file.write(result_ce_ft_2)

    end_time = time.time()
    search_time = end_time - start_time
    print(f"Execution time: {search_time}")


# Terminal Command: python3 Main.py Answers.json topics_1.json
# OR python3 Main.py Answers.json topics_2.json
if __name__ == "__main__":
    # Ensure two arguments are passed (answers.json and topics.json)
    if len(sys.argv) != 4:
        print("Usage: python main.py <answers.json> <topics_1.json> <topics_2.json>")
        sys.exit(1)
    # Get file paths from command line arguments
    answers_file = sys.argv[1]
    topics_1_file = sys.argv[2]
    topics_2_file = sys.argv[3]

    # Call the main function with the file paths
    main(answers_file, topics_1_file, topics_2_file)

# Run the retrieval methods to get the results files.
# Version 31.10.2024

import sys
import json
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
import BiEncoder

# Function to create QREL files
def create_qrel_files():
    BiEncoder.create_qrel_files()

def load_models():
    pretrained_bi_model = SentenceTransformer('multi-qa-distilbert-cos-v1')
    pretrained_cross_model = CrossEncoder('cross-encoder/stsb-distilroberta-base')
    return pretrained_bi_model, pretrained_cross_model

def load_qrels():
    train_qrel = json.load(open('train_qrel.json'))
    val_qrel = json.load(open('val_qrel.json'))
    return train_qrel, val_qrel

def load_queries(topics_file):
    dic_topics = BiEncoder.load_topic_file(topics_file)
    return {qid: "[TITLE]" + text[0] + "[BODY]" + text[1] for qid, text in dic_topics.items()}

def fine_tune_models(train_qrel, val_qrel):
    ft_bi_encoder_model = BiEncoder.train(SentenceTransformer('multi-qa-distilbert-cos-v1'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ft_cross_encoder_model = MyCrossEncoder.finetune(CrossEncoder('cross-encoder/stsb-distilroberta-base', device=device), train_qrel, val_qrel)
    return ft_bi_encoder_model, ft_cross_encoder_model

def run_bi_encoder_retrievals(model, queries, collection_dic, output_prefix):
    result = Retrievals.bi_retrieve(model, queries, collection_dic)
    with open(f"{output_prefix}.tsv", "w") as file:
        for query_id, results in result.items():
            for doc_id, score in results:
                file.write(f"{query_id}\t{doc_id}\t{score}\n")

def run_cross_encoder_retrievals(model, queries, collection_dic, result_prefix):
    result_file_path = f"{result_prefix}_bi_results.tsv"
    result = Retrievals.cross_retrieval(model, queries, collection_dic, result_file_path)
    with open(f"{result_prefix}.tsv", "w") as file:
        for query_id, doc_id, score in result:
            file.write(f"{query_id}\t{doc_id}\t{score}\n")

def main(answers, topics_1, topics_2):
    print("Starting main...")
    pretrained_bi_model, pretrained_cross_model = load_models()
    train_qrel, val_qrel = load_qrels()
    collection_dic = BiEncoder.read_collection(answers)

    queries_1 = load_queries(topics_1)
    queries_2 = load_queries(topics_2) if topics_2 else {}

    ft_bi_encoder_model, ft_cross_encoder_model = fine_tune_models(train_qrel, val_qrel)

    # Load test QRELs
    with open('test_qrel.json', 'r') as f:
        test_dic_qrel = json.load(f)

    # Perform retrievals
    run_bi_encoder_retrievals(pretrained_bi_model, test_dic_qrel, collection_dic, "result_bi_1")
    if queries_2:
        run_bi_encoder_retrievals(pretrained_bi_model, queries_2, collection_dic, "result_bi_2")

    run_bi_encoder_retrievals(ft_bi_encoder_model, test_dic_qrel, collection_dic, "result_bi_ft_1")
    if queries_2:
        run_bi_encoder_retrievals(ft_bi_encoder_model, queries_2, collection_dic, "result_bi_ft_2")

    run_cross_encoder_retrievals(pretrained_cross_model, test_dic_qrel, collection_dic, "result_ce_1")
    if queries_2:
        run_cross_encoder_retrievals(pretrained_cross_model, queries_2, collection_dic, "result_ce_2")

    run_cross_encoder_retrievals(ft_cross_encoder_model, test_dic_qrel, collection_dic, "result_ce_ft_1")
    if queries_2:
        run_cross_encoder_retrievals(ft_cross_encoder_model, queries_2, collection_dic, "result_ce_ft_2")

# Terminal Command: python3 Main.py Answers.json topics_1.json
# OR python3 Main.py Answers.json topics_2.json
if __name__ == "__main__":
    create_qrel_files()
    import MyCrossEncoder
    import Retrievals

    if len(sys.argv) != 4:
        print("Usage: python main.py <answers.json> <topics_1.json> <topics_2.json>")
        sys.exit(1)

    answers_file = sys.argv[1]
    topics_1_file = sys.argv[2]
    topics_2_file = sys.argv[3]

    main(answers_file, topics_1_file, topics_2_file)
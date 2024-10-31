# Run the retrieval methods to get the results files. Below is from assignment 3 pdf:
# You will get 8 retrieval result files by the end of this assignment. You should submit them all
# following the naming convention below:
# ● result_bi_1.tsv: bi-encoder with no fine-tuning on your test set
# ● result_bi_2.tsv: bi-encoder with no fine-tuning on topic_2 file
# ● result_bi_ft_1.tsv: bi-encoder with fine-tuning on your test set
# ● result_bi_ft_2.tsv: bi-encoder with fine-tuning on topic_2 file
# ● result_ce_1.tsv: cross-encoder with no fine-tuning on your test set
# ● result_ce_2.tsv: cross-encoder with no fine-tuning on topic_2 file
# ● result_ce_ft_1.tsv: cross-encoder with fine-tuning on your test set
# ● result_ce_ft_2.tsv: cross-encoder with fine-tuning on topic_2 file
import sys
import time
import Retrievals
from sentence_transformers import SentenceTransformer, CrossEncoder
import json
import BiEncoder


def main(topics, answers):
    print("Starting...")
    start_time = time.time()

    with open(topics, 'r') as f:
        queries = json.load(f)

    with open(answers, 'r') as f:
        docs = json.load(f)

    # using test set created in BiEncoder.py
    with open('test_qrel.json', 'r') as f:
        test_dic_qrel = json.load(f)

    if topics == 'topics_2.json':
        with open('topics_2.json', 'r') as f:
            queries_2 = json.load(f)

    # load each pretrained model
    pretrained_bi_model = SentenceTransformer('sentence-transformers/multi-qa-distilbert-cos-v1')
    pretrained_cross_model = CrossEncoder('cross-encoder/stsb-distilroberta-base'
                                          )
    # train model and get MODEL
    model_path = BiEncoder.train(pretrained_bi_model)
    # model_path = need one for cross encoder but not coded in functions, fix first
    ft_bi_encoder_model = SentenceTransformer(model_path)

    # Call retrieval methods and output result files
    # 1. No fine-tuning on test set: bi-encoder
    result_bi_1 = Retrievals.bi_retrieve(pretrained_bi_model, test_dic_qrel, docs)
    with open("result_bi_1.tsv", "w") as file:
        file.write(result_bi_1)

    # 2. No fine-tuning on topics_2 file: bi-encoder
    result_bi_2 = Retrievals.bi_retrieve(pretrained_bi_model, queries_2, docs)
    with open("result_bi_2.tsv", "w") as file:
        file.write(result_bi_2)

    # 3. Fine-tuning on test set: bi-encoder
    result_bi_ft_1 = Retrievals.bi_retrieve(ft_bi_encoder_model, test_dic_qrel, docs)
    with open("result_bi_ft_1.tsv", "w") as file:
        file.write(result_bi_ft_1)

    # 4. Fine-tuning on topics_2 file: bi-encoder
    result_bi_ft_2 = Retrievals.bi_retrieve(ft_bi_encoder_model, queries_2, docs)
    with open("result_bi_ft_2.tsv", "w") as file:
        file.write(result_bi_ft_2)

    # 5. No fine-tuning on test set: cross-encoder
    # result_ce_1 = Retrievals.cross_retrieval()

    # 6. No fine-tuning on topics_2 file: cross-encoder

    # 7. Fine-tuning on test set: cross-encoder

    # 8. Fine-tuning on topics_2 file: cross-encoder


    end_time = time.time()
    print(f"Execution time: {end_time-start_time}")


# Terminal Command: python Main.py Answers.json topics_1.json
# OR python Main.py Answers.json topics_2.json
if __name__ == "__main__":
    # Ensure two arguments are passed (answers.json and topics.json)
    if len(sys.argv) != 3:
        print("Usage: python main.py <answers.json> <topics.json>")
        sys.exit(1)

    # Get file paths from command line arguments
    answers_file = sys.argv[1]
    topics_file = sys.argv[2]

    # Call the main function with the file paths
    main(answers_file, topics_file)

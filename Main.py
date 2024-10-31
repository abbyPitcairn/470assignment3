# Run the retrieval methods to get the results files.
import sys
import time
import Retrievals
from sentence_transformers import SentenceTransformer, util


def main(topics, answers):
    print("Starting main...")

    # Initialize bi-encoder base model
    bi_model = SentenceTransformer('sentence-transformers/multi-qa-distilbert-dot-v1')

    # Initialize cross-encoder base model
        # Implement here

    # Conduct all eight searches
    # Input: model, topics filepath, answers filepath, output filepath
    # ● result_bi_1.tsv: bi-encoder with no fine-tuning on your test set
    bi_search(bi_model, topics, answers, 'result_bi_1.tsv')

    # ● result_bi_2.tsv: bi-encoder with no fine-tuning on topic_2 file

    # ● result_bi_ft_1.tsv: bi-encoder with fine-tuning on your test set

    # ● result_bi_ft_2.tsv: bi-encoder with fine-tuning on topic_2 file

    # ● result_ce_1.tsv: cross-encoder with no fine-tuning on your test set

    # ● result_ce_2.tsv: cross-encoder with no fine-tuning on topic_2 file

    # ● result_ce_ft_1.tsv: cross-encoder with fine-tuning on your test set

    # ● result_ce_ft_2.tsv: cross-encoder with fine-tuning on topic_2 file



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


# Streamline the bi-encoder retrieval process and record search execution time
# Model: the model to pass to bi-retrieval
# Topics: the filepath to the queries
# Answers: the filepath to the documents
# Output_filepath: the name of the output results file
def bi_search(model, topics, answers, output_filepath):
    start_time = time.time()
    bi_result = Retrievals.bi_retrieve(model, topics, answers)
    Retrievals.save_to_result_file(bi_result, output_filepath)
    end_time = time.time()
    search_time = end_time - start_time
    print(f"Execution time for bi-encoder retrieval with {model}: {search_time}")


# Streamline the cross-encoder retrieval process and record search execution time
# Model: the model to pass to cross-retrieval
# Topics: the filepath to the queries
# Answers: the filepath to the documents
# Output_filepath: the name of the output results file
def cross_search(model, topics, answers, output_filepath):
    start_time = time.time()
    cross_result = Retrievals.bi_retrieve(model, topics, answers)
    Retrievals.save_to_result_file(cross_result, output_filepath)
    end_time = time.time()
    search_time = end_time - start_time
    print(f"Execution time: {search_time}")

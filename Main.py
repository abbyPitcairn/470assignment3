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


def main(topics, answers):
    print("Starting...")
    start_time = time.time()
    # Call retrieval methods and output result files
    end_time = time.time()
    print(f"Execution time: {end_time-start_time}")


# Terminal Command: python Main.py Answers.json topics_1.json
# OR python Main.py Answers.json topics_2.json
if __name__ == "__main__":
    # Ensure two arguments are passed (answers.json and topics.json)
    if len(sys.argv) != 2:
        print("Usage: python main.py <answers.json> <topics.json>")
        sys.exit(1)

    # Get file paths from command line arguments
    answers_file = sys.argv[1]
    topics_file = sys.argv[2]

    # Call the main function with the file paths
    main(answers_file, topics_file)

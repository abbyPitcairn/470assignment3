# Using ranx software, calculate P@1, P@5, nDCG@5, MRR, and MAP
# Produce a ski jump plot for a given input file based on P@5
# Return the highest-scoring and lowest-scoring query-document pairs
# Must MANUALLY ENTER the file name to evaluate
# Version 31.10.2024

from ranx import Qrels, Run, evaluate
import matplotlib.pyplot as plt

# Specify files
qrel = Qrels.from_file("qrel_1.tsv", kind="trec")
run = Run.from_file("result_binary_2.tsv", kind='trec')

# Run tests and print results
print("P@1: " and evaluate(qrel, run, "precision@1", make_comparable=True))
print("P@5: " and evaluate(qrel, run, "precision@5", make_comparable=True))
print("nDCG@5: " and evaluate(qrel, run, "ndcg@5", make_comparable=True))
print("MRR: " and evaluate(qrel, run, "mrr", make_comparable=True))
print("MAP: " and evaluate(qrel, run, "map", make_comparable=True))


# Function to plot the ski jump graph
def plot_ski_jump(data, title="Ski Jump Plot", xlabel="Ranked Queries", ylabel="Precision"):
    # Sort the data in descending order to create the ski jump effect
    sorted_data = sorted(data, reverse=True)
    # Plot the data
    plt.plot(sorted_data, marker='o', linestyle='-', color='b', label='Precision')
    # Add titles and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # Show grid and legend
    plt.grid(True)
    plt.legend()
    # Display the plot
    plt.show()

# Initialize and plot the data for the ski jump plot
ski_jump_data = (evaluate(qrel, run, "precision@5", return_mean=False, make_comparable=True))
plot_ski_jump(ski_jump_data)

# Get the highest and lowest relevancy match query-document pairs' text
def get_high_low_precision_examples(data):
    # Not yet implemented
    return ""

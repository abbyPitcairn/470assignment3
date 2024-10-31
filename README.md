# 470 Assignment 3

### Assignment by Mandy Ho and Abby Pitcairn
#### Version: 31 October 2024

Comparing bi-encoder and cross-encoder BERT IR systems with and without finetuning. 

Base model for bi-encoder: 'multi-qa-distilbert-cos-v1'

Base model for cross-encoder: 'cross-encoder/stsb-distilroberta-base'

Models' documentation: https://sbert.net/docs/sentence_transformer/pretrained_models.html


### Command Line
Running Main.py takes three inputs: Answers.json and topics_1.json and topics_2.json.
Example command: python3 Main.py Answers.json topics_1.json topics_2.json
All eight result files can be generated from this command.

### Results

Output will be eight result files corresponding to:

- Bi-encoder retrieval on topics 1
- Bi-encoder retrieval on topics 2
- Bi-encoder with fine-tuning on topics 1
- Bi-encoder with fine-tuning on topics 2
- Cross-encoder retrieval on topics 1
- Cross-encoder retrieval on topics 2
- Cross-encoder with fine-tuning on topics 1
- Cross-encoder with fine-tuning on topics 2

Evaluation.py will evaluate output files for precision@1, precision@5, nDCG@5, MRR, MAP. 
Ski-jump plot is created for P@5. 
Highest and lowest scoring document-query pairs are returned for analysis.

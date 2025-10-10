import argparse
import json
from data_processing import main as data_processing_main
from moral_word_prediction_training import main as training_main
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from embeddings_analysis.correlation_analysis import main as embeddings_analysis_main
from sentence_embeddings.generate_sentence_embeddings_1 import main as generate_sentence_embeddings_main

def main():
    parser = argparse.ArgumentParser(description="Main script for moral word prediction training.")

    parser.add_argument(
        "--generate-embeddings-args",
        type=str,
        help="Generate sentence embeddings for characters in movies.",
        required=False
    )

    parser.add_argument(
        "--data-processing-args",
        type=str,
        help="Arguments for the data processing step, passed as a JSON string.",
        required=False
    )

    parser.add_argument(
        "--training-args",
        type=str,
        help="Arguments for the training step, passed as a JSON string.",
        required=False
    )

    parser.add_argument(
        "--embeddings-analysis-args",
        type=str,
        help="Arguments for the embeddings analysis step, passed as a JSON string.",
        required=False
    )

    args = parser.parse_args()
    

    # Parse arguments for embeddings generation
    if args.generate_embeddings_args:
        generate_embeddings_args = json.loads(args.generate_embeddings_args)
        generate_sentence_embeddings_main(argparse.Namespace(**generate_embeddings_args))

    # Parse arguments for data processing
    if args.data_processing_args:
        data_processing_args = json.loads(args.data_processing_args)
        data_processing_main(argparse.Namespace(**data_processing_args))

    # Parse arguments for training and evaluation
    if args.training_args:
        training_args = json.loads(args.training_args)
        training_main(argparse.Namespace(**training_args))
    
    # Parse arguments for embeddings analysis
    if args.embeddings_analysis_args:
        embeddings_analysis_args = json.loads(args.embeddings_analysis_args)
        embeddings_analysis_main(argparse.Namespace(**embeddings_analysis_args))


if __name__ == "__main__":
    main()
 
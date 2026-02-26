import joblib
import numpy as numpy
import bm25s
import os
from sentence_transformers import SentenceTransformer

from utils import (
    read_dataframe,
    pprint,
    generate_with_single_input,
    cosine_similarity,
    display_widget,
)

NEWS_DATA = read_dataframe("news_data_dedup.csv")


def query_news(indices):
    """
    Retrieves elements from a dataset based on specified indices.

    Parameters:
    indices (list of int): A list containing the indices of the desired elements in the
    dataset.
    dataset(list or sequence): The dataset from which elements are to be retrieved.
    It should support indexing.

    Returns:
    list: A list of elements from the dataset corresponding to the indices provided in
    list_of_indices.
    """

    output = [NEWS_DATA[index] for index in indices]
    return output


# The corpus used will be the title appended with the description
corpus = [x["title"] + " " + x["description"] for x in NEWS_DATA]

# Instantiate the retriever by passing the corpus data
BM25_RETRIEVER = bm25s.BM25(corpus=corpus)

# Tokenise the chunks
tokenised_data = bm25s.tokenize(corpus)

# Index the tokenised chunks within this retriever
BM25_RETRIEVER.index(tokenised_data)

# Tokenise the query
sample_query = "What are the recent news about GDP?"
tokenised_sample_query = bm25s.tokenize(sample_query)

# Get the retrieved results and their respective scores
results, scores = BM25_RETRIEVER.retrieve(tokenised_sample_query, k=3)

print(f"Results for query: {sample_query}\n")

for doc in results[0][0]:
    print(f"Document retrieved {corpus.index(doc)} : {doc}")

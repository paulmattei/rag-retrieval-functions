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


def bm25_retrieve(query: str, top_k: int = 5):
    """
    Retrieves the top k relevant documents for a given query using the BM25 algorithm.

    This function tokenises the input query and uses a pre-indexed BM25 reciever to
    search through a collection of documents. It returns the indices of the top k
    documents that are most relevant to the the query.

    Args:
        query (str): The search query for which the documents need to be retrieved.
        top_k (int): The number of top relevant documents to retrieve. Default is 5.

    Returns:
        List[int]: A list of indices corresponding to the top k relevant documents
        within the corpus.
    """

    # Tokenise the query using the 'tokenize' function from the bm25s module
    tokenised_query = bm25s.tokenize(query)

    # Use the BM25_RETRIEVER to retrive documents and their scores based on the
    # tokenised query. Retrieve the top k documents
    results, scores = BM25_RETRIEVER.retrieve(tokenised_query, k=top_k)

    # Extract the first element to from results to get the list of retrieved documents
    results = results[0]

    top_k_indices = [corpus.index(result) for result in results]

    return top_k_indices


if __name__ == "__main__":
    bm25_retrieve("What about GDP?")

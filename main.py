import joblib
import numpy as np
import bm25s
import os
import transformers
from sentence_transformers import SentenceTransformer

transformers.logging.set_verbosity_error()

from utils import (
    read_dataframe,
    pprint,
    generate_with_single_input,
    cosine_similarity,
    display_widget,
)


def query_news(indices, NEWS_DATA):
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


def bm25_retrieve(query: str, bm25_retriever, corpus, top_k: int = 5):
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

    # Use the bm25_retriever to retrive documents and their scores based on the
    # tokenised query. Retrieve the top k documents
    results, scores = bm25_retriever.retrieve(tokenised_query, k=top_k)

    # Extract the first element to from results to get the list of retrieved documents
    results = results[0]

    top_k_indices = [corpus.index(result) for result in results]

    return top_k_indices


def semantic_search_retrieve(
    query,
    model,
    embeddings,
    top_k=5,
):
    """
    Retrieves the top k relevant documents for a given query using semantic search and
    cosine similarity.
    This function generates an embedding for the input query and compares it against
    precomputed document embeddings using cosine similarity. The indices of the top k
    most similar documents are returned.

    Args:
        query (str): The search query for which the documents need to be retrieved.
        top_k (int): The number of top relevant documents to retrieve. Default is 5.

    Returns:
        List[int]: A list of indices corresponding to the top k relevant documents
        within the corpus.
    """

    # Generate the embedding for the query using the pretrained model
    query_embedding = model.encode(query)

    # Calculate the cosine similarity scores in descending order to get the indices
    similarity_scores = cosine_similarity(query_embedding, embeddings)

    # Sort the similarity scores in descending order and get the indices
    similarity_indices = np.argsort(-similarity_scores)

    # Select the indices of the top k documents as a numpy array
    top_k_indices_array = similarity_indices[:top_k]

    # Cast them to int
    top_k_indices = [int(x) for x in top_k_indices_array]

    return top_k_indices


def reciprocal_rank_fusion(list1, list2, top_k=5, K=60):
    """
    Fuse rank from multiple IR systems using Reciprocal Rank Fusion.

    Args:
        list1 (list[int]): A list of indices of the the top k documents that match the
                           query.
        list2 (list[int]): Another list of indices of the top k dociments that match the
                           query.
        top_k (int): The number of documents to concider from each list for fusion.
                     Defaults to 5.
        K (int): A constant used in the RR formula. Defaults to 60.

    Returns:
        list[int]: A list of indices of the top k documents sorted by their RRF scores.
    """

    # Create a dictionary to store the RRF score for each document index.
    rrf_scores = {}

    # Iterate over each document list
    for lst in [list1, list2]:
        # Calculate the RRF score for each document index
        for rank, item in enumerate(
            lst, start=1
        ):  # Sets first element as 1 not 0, as is convention.
            # If the item is not in the dictionary, initialize it's score to 0
            if item not in rrf_scores:
                rrf_scores[item] = 0
            # Update the RRF score for each document index using the formula 1 / (rank + K)
            rrf_scores[item] += 1 / (rank + K)

    # Sort the document indices based on their RRF scores in decending order
    sorted_items = sorted(rrf_scores.keys(), key=rrf_scores.get, reverse=True)

    # Slice the list to get the top-k document indices
    top_k_indices = [int(x) for x in sorted_items[:top_k]]

    return top_k_indices


def generate_final_prompt(
    query,
    top_k,
    model,
    embeddings,
    bm25_retriever,
    corpus,
    news_data,
    retrieve_function=None,
    use_rag=True,
):
    """
    Generates an augmented prompt for a Retrieval Augmented Generation system by
    retrieving the top_k most relevant documents based on a given query.

    Args:
        query (str): The search query for which the relevant documents are to be
                     retrieved.
        top_k (int): The number of top relevant documents to retrieve.
        retrieve_function (callable): The function used to retrieve relevant documents.
                                      If 'reciprocal_rank_fusion', it will combine the
                                      the results from different retrieval functions.
        use_rag (bool): A flag to determine whether to incorporate retrieved data into
                        the prompt (default is True).

        Returns:
        str: A prompt that includes the top_k relevant documents formatted for use in
             a RAG system.
    """

    # Define the prompt as the initial query
    prompt = query

    # If not using RAG, return the prompt
    if not use_rag:
        return prompt

    # Determine which retrieve function to use based on it's name.
    if retrieve_function.__name__ == "reciprocal_rank_fusion":
        # Retrieve top documents using two different methods.
        list1 = semantic_search_retrieve(query, model, embeddings, top_k)
        list2 = bm25_retrieve(query, bm25_retriever, corpus, top_k)
        # Combine the results using reciprocal rank fusion.
        top_k_indices = retrieve_function(list1, list2, top_k)
    else:
        # Use the provided retrieval function.
        top_k_indices = retrieve_function(query=query, top_k=top_k)

    # Retrieve documents from the dataset using the indices.
    relevant_documents = query_news(top_k_indices, news_data)

    formatted_documents = []

    for document in relevant_documents:
        formatted_document = (
            f"Title: {document['title']}, Description: {document['description']}, "
            f"Published at: {document['published_at']}\nURL: {document['url']}"
        )

        # Append the formatted string to the main data string with a newline
        # for seperation.
        formatted_documents.append(formatted_document)

    retrieve_data_formatted = "\n".join(formatted_documents)

    prompt = (
        f"Answer the user query below. There will be provided additional information "
        f"for you to compose your answer. The relevant infomration provided is from "
        f"2024 and it should be added as your overall knowledge to answer the query, "
        f"you should not rely only on this infomration to answer the query, but add it "
        f"to your overall knowledge.\n"
        f"Query: {query}\n"
        f"2024 News: {retrieve_data_formatted}"
    )

    return prompt


def llm_call(
    query,
    model,
    embeddings,
    bm25_retriever,
    corpus,
    news_data,
    retrieve_function=None,
    top_k=5,
    use_rag=True,
):
    # Get the system and user dictionaires
    prompt = generate_final_prompt(
        query,
        top_k=top_k,
        model=model,
        embeddings=embeddings,
        bm25_retriever=bm25_retriever,
        corpus=corpus,
        news_data=news_data,
        retrieve_function=retrieve_function,
        use_rag=use_rag,
    )
    generated_response = generate_with_single_input(prompt)
    generated_message = generated_response["content"]
    return generated_message


def main():

    news_data = read_dataframe("news_data_dedup.csv")

    # The corpus used will be the title appended with the description
    corpus = [x["title"] + " " + x["description"] for x in news_data]

    # Instantiate the retriever by passing the corpus data
    bm25_retriever = bm25s.BM25(corpus=corpus)

    # Tokenise the chunks
    tokenised_data = bm25s.tokenize(corpus)

    # Index the tokenised chunks within this retriever
    bm25_retriever.index(tokenised_data)

    embeddings = joblib.load("embeddings.joblib")

    model_name = "models/BAAI/bge-base-en-v1.5"
    model = SentenceTransformer(model_name)

    query = "What do we know about GDP?"
    #    keyword_list = bm25_retrieve(query)
    #    semantic_list = semantic_search_retrieve(query)
    #    reciprocal_rank_fusion(keyword_list, semantic_list)
    print(
        llm_call(
            query,
            model,
            embeddings,
            bm25_retriever,
            corpus,
            news_data,
            retrieve_function=reciprocal_rank_fusion,
        )
    )


if __name__ == "__main__":
    main()

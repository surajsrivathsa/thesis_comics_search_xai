import os, sys
import random
import torch
from sentence_transformers import SentenceTransformer, util
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import common_functions.backend_utils as utils

# book_metadata_dict, comic_book_metadata_df = utils.load_book_metadata()
book_cover_prompt_dict = utils.load_local_explanation_book_cover()


def create_model():
    sentence_transformer_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    print(sentence_transformer_model.encode(["test sentence"]))
    print()
    print(" ================== ================ ================= ")
    print("done loading model")
    print("book cover prompt {}".format(book_cover_prompt_dict[2]))
    print()
    print(" ================== ================ ================= ")
    return sentence_transformer_model


def shutdown_model(sentence_transformer_model):
    del sentence_transformer_model


def create_docs_lst(book_ids_lst: list, prompt_pkl_dict: dict, sample_size=10):
    """
    """
    book_docs_lst = []
    for book_id in book_ids_lst:
        for doc in prompt_pkl_dict.get(book_id, "comic book cover"):
            book_docs_lst.extend(
                [
                    sentence.strip()
                    for sentence in doc.split(",")
                    if utils.filter_fnc(sentence)
                ]
            )

    if len(book_docs_lst) > sample_size and sample_size > 0:
        book_docs_lst = random.sample(book_docs_lst, sample_size)

    return book_docs_lst


def best_matching_themes_between_interestedbooks_and_searchresults(
    corpus, queries, sim_cutoff=0.8, model=None, return_top_k=10
):
    # encode corpus
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

    # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    top_k = min(1, len(corpus))
    count = 0
    similarity_between_interested_books_and_search_results_lst = []

    for query in queries:
        query_embedding = model.encode(query, convert_to_tensor=True)

        # We use cosine-similarity and torch.topk to find the highest 5 scores
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        for score, idx in zip(top_results[0], top_results[1]):
            # print(corpus[idx], "(Score: {:.4f})".format(score))
            if score < sim_cutoff:
                similarity_between_interested_books_and_search_results_lst.append(
                    (query, corpus[idx], score.numpy().item())
                )

        count += 1

    sorted_explanations_tup_lst = sorted(
        similarity_between_interested_books_and_search_results_lst,
        key=lambda tup: (tup[2], len(tup[0]), len(tup[1])),
        reverse=True,
    )
    sorted_explanations_lst = list_of_lists = [
        list(elem) for elem in sorted_explanations_tup_lst
    ]
    return sorted_explanations_lst[:return_top_k]


async def explain_relevance_feedback(
    clicksinfo_dict: list, query_book_id: int, search_results: list, model
):
    """
    """
    loop = asyncio.get_running_loop()
    interested_book_ids_lst = [query_book_id] + [
        book["comic_no"] for book in clicksinfo_dict if book["interested"] == 1.0
    ]
    search_results_book_ids_lst = [book["comic_no"] for book in search_results]

    # interested_book_docs_lst = await loop.create_docs_lst(
    #     None, create_docs_lst, interested_book_ids_lst, book_cover_prompt_dict, 50
    # )

    interested_book_docs_lst = create_docs_lst(
        interested_book_ids_lst, prompt_pkl_dict=book_cover_prompt_dict, sample_size=50
    )

    # search_results_book_docs_lst = await loop.create_docs_lst(
    #     None, create_docs_lst, search_results_book_ids_lst, book_cover_prompt_dict, 300
    # )
    search_results_book_docs_lst = create_docs_lst(
        search_results_book_ids_lst,
        prompt_pkl_dict=book_cover_prompt_dict,
        sample_size=300,
    )

    top_k_matching_themes_tup_lst = best_matching_themes_between_interestedbooks_and_searchresults(
        corpus=search_results_book_docs_lst,
        queries=interested_book_docs_lst,
        model=model,
        return_top_k=10,
    )

    # top_k_matching_themes_tup_lst = await loop.create_docs_lst(
    #     None,
    #     best_matching_themes_between_interestedbooks_and_searchresults,
    #     corpus=search_results_book_docs_lst,
    #     queries=interested_book_docs_lst,
    #     sim_cutoff=0.8,
    #     model=model,
    #     return_top_k=10,
    # )

    return {"relevance_feedback_explanation": [sublst[0] for sublst in top_k_matching_themes_tup_lst]}


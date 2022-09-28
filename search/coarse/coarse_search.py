import os, sys
import pandas as pd, numpy as np
import re
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.dirname(__file__))) )

## import custom functions
import common_functions.backend_utils as utils
import common_constants.backend_constants as cst

## Loading features and metadata beforehand
cld_tfidf_df, cld_tfidf_np, edh_tfidf_df, edh_tfidf_np, hog_tfidf_df, hog_tfidf_np, text_tfidf_df, text_tfidf_np = utils.load_all_coarse_features()
book_metadata_dict, comic_book_metadata_df = utils.load_book_metadata()


def get_top_n_matching_book_info(idx_top_n_np, sim_score_top_n_np, comic_info_dict=book_metadata_dict, print_n=20, query_book_id=1, feature_similarity_type='cld'):
  sim_score_top_n_squeezed_np = np.squeeze(sim_score_top_n_np)
  list_of_records = []
  query_comic_no, query_book_title, query_genre = comic_info_dict[query_book_id]
  
  for i in range(1, print_n):
    
    book_idx = idx_top_n_np[i]
    sim_score_book = sim_score_top_n_squeezed_np[i]

    try:
      comic_no, book_title, genre = comic_info_dict[book_idx]
    except Exception as e:
      comic_no, book_title, genre = (-1, 'not exist', 'not exist')
      
    list_of_records.append({'rank': i, 'sim_score': sim_score_book, 'comic_no': comic_no, 'book_title': book_title, 'genre': genre, 
                            'query_comic_no':query_comic_no, 'query_book_title':query_book_title, 'query_genre':query_genre, 'feature_similarity_type': feature_similarity_type})

  df = pd.DataFrame.from_dict(list_of_records)
  return df


def comics_coarse_search(query_comic_book_id: int, feature_weight_dict: dict, top_n: int):

    # remove this later
    query_book_id = query_comic_book_id - 3451
    
    # get similarity for all features
    cld_cosine_similarity = utils.cosine_similarity(cld_tfidf_np[:, :], cld_tfidf_np[max(query_book_id, 0):query_book_id+1,  : ])
    edh_cosine_similarity = utils.cosine_similarity(edh_tfidf_np[:, :], edh_tfidf_np[max(query_book_id, 0):query_book_id+1,  : ])
    hog_cosine_similarity = utils.cosine_similarity(hog_tfidf_np[:, :], hog_tfidf_np[max(query_book_id, 0):query_book_id+1,  : ])
    text_cosine_similarity = utils.cosine_similarity(text_tfidf_np[:, :], text_tfidf_np[max(query_book_id, 0):query_book_id+1,  : ])
    
    # combine similarity and weigh them
    combined_results_similarity = cld_cosine_similarity *  feature_weight_dict['cld'] + edh_cosine_similarity *  feature_weight_dict['edh'] + hog_cosine_similarity *  feature_weight_dict['hog'] + text_cosine_similarity *  feature_weight_dict['text'] 
    
    # find top book indices according to combined similarity
    combined_results_indices = np.argsort(np.squeeze(-combined_results_similarity), axis=0)
    
    # sort indices by their combined similarity score to pick top k
    combined_sorted_result_indices = np.sort(-combined_results_similarity,axis=0 )
    
    top_k_df = get_top_n_matching_book_info(idx_top_n_np=combined_results_indices, sim_score_top_n_np=combined_sorted_result_indices, comic_info_dict=book_metadata_dict, print_n=top_n, query_book_id=query_book_id, feature_similarity_type= 'coarse_combined')
    
    return top_k_df



if __name__ == '__main__':
    query_book_comic_id = 3480
    top_n = 21
    feature_weight_dict = {'cld': 0.2, 'edh': 0.2, 'hog': 0.2, 'text': 1.4} #{'cld': 0.4, 'edh': 0.4, 'hog': 0.4, 'text': 0.8}
    
    print('query book info: {}'.format(book_metadata_dict[query_book_comic_id-3451]))
    
    top_n_results_df = comics_coarse_search(query_book_comic_id, feature_weight_dict=feature_weight_dict, top_n=top_n)
    print(top_n_results_df.head(top_n))
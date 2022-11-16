import pandas as pd, numpy as np
import os, sys
from sklearn.cluster import KMeans
import pickle
from multiprocessing import Pool
import time
import math
import glob
import warnings
warnings.filterwarnings('ignore')

def read_descriptor_data(filepath: str, filetype: str):
    
    if filetype == 'csv':
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)
    
    return df


def extract_book_metadata(filepath:str, filetype: str):
    
    if filetype == 'csv':
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)
        
    df['idx'] = df.index.copy()
    
    grouped_comic_df = df.groupby(['comic_id'])['idx'].agg(book_start_idx='min', book_end_idx='max').reset_index()
    grouped_comic_df['book_end_idx'] = grouped_comic_df['book_end_idx'] + 1
    
    book_metadata_lst = list(grouped_comic_df.to_numpy().tolist())
    
    return book_metadata_lst


def extract_book_metadata_from_folderpaths(parent_dir: str, image_format: str):
    counter = 0
    comic_id = 3949
    prev_comic_book_title = ''
    book_metadata_lst = []
    records_lst = []
    for f in glob.glob(os.path.join(parent_dir, "**", "*." + image_format), recursive=True ):
        
        comic_book_title = os.path.basename(os.path.dirname(f))
        idx = counter
        
        if prev_comic_book_title != comic_book_title:
            comic_id += 1
            prev_comic_book_title = comic_book_title
            
        records_lst.append({ 'comic_id': comic_id, 'idx': idx, 'comic_book_title': comic_book_title })
        counter += 1
        
    df = pd.DataFrame.from_records(records_lst)
    # df.to_excel('df.xlsx', index=False)
    print(df.shape)
    print(df.head(10))
    grouped_comic_df = df.groupby(['comic_id'])['idx'].agg(book_start_idx='min', book_end_idx='max').reset_index()
    grouped_comic_df['book_end_idx'] = grouped_comic_df['book_end_idx'] + 1
    book_metadata_lst = list(grouped_comic_df.to_numpy().tolist())
    
    return book_metadata_lst


def build_histogram(descriptor_np, cluster_alg):
    histogram = list(np.zeros(len(cluster_alg.cluster_centers_)))
    cluster_result =  cluster_alg.predict(descriptor_np)
    for i in cluster_result:
        histogram[i] += 1.0
    return histogram


def build_visual_words(descriptor_np: np.array, book_metadata_lst: list, num_visual_words: int, output_folder_path: str, descriptor_type: str, model: any):

    start_time = time.time()
    print('start time for {}: {}'.format(descriptor_type, start_time))
    
    # load provided model else generate new one
    if model:
        kmeans = model
    else:
        
        kmeans = KMeans(n_clusters = num_visual_words)
        kmeans.fit(descriptor_np)
    
    # save model
    with open(os.path.join(output_folder_path, "kmeans_{}.pickle".format(descriptor_type) ), "wb") as handle:
        pickle.dump(kmeans, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    kmeans_time = time.time()
    print('kmeans completed for {} at : {} min from start'.format( descriptor_type, (kmeans_time-start_time)/60.0 ))
    
    # for us , a visual word equals every image and a book is comprised of as many words as images, so we build feature per book
    codebook_lst = []   
    for idx, sub_lst in enumerate(book_metadata_lst):
        
        # initialize an empty list for record
        record_lst = []
        comic_id = sub_lst[0]
        record_lst.append(comic_id)
        
        # choose descriptors for specific book
        book_descriptor_np = descriptor_np[ sub_lst[1]: sub_lst[2], : ]
        
        # build histogram
        histogram = build_histogram(book_descriptor_np, kmeans)
        record_lst.extend(histogram)
        
        # add book information to codebook
        codebook_lst.append(record_lst)
    
    codebook_time = time.time()    
    print('codebook completed for {} at : {} min from start'.format( descriptor_type, (codebook_time-kmeans_time)/60.0 ))
    
    col_name_lst = ['comic_id'] + ['vw_{}'.format(str(idx)) for idx in range(num_visual_words)]
    codebook_df = pd.DataFrame(columns=col_name_lst, data=codebook_lst)
    codebook_df.to_csv( os.path.join(output_folder_path, '{}_codebook_df.csv'.format(descriptor_type) ), index=False)
    
    print('process completed for {} at : {} min from start'.format( descriptor_type, (time.time()-codebook_time)/60.0 ))
    
    return True


def find_tf_idf_of_bovw(histogram_df: pd.DataFrame):
    
    feature_df = histogram_df.copy().drop(columns='comic_id', axis=1)
    feature_cols_lst = list(feature_df.columns)
    feature_df['total_words_in_document'] =feature_df.sum(numeric_only=True, axis=1)
    feature_df[feature_cols_lst] = feature_df[feature_cols_lst].div(feature_df['total_words_in_document'], axis=0)
    
    inverse_document_frequency_dict = {feature_col:1e-6 for feature_col in feature_cols_lst}
    tf_idf_df = pd.DataFrame()
    
    for col_name in feature_cols_lst:
        document_frequency = feature_df[feature_df[col_name] > 0.0].shape[0]
        # print('{}: {}'.format(col_name, document_frequency))
        log_inverse_document_frequency = math.log(feature_df.shape[0]/(document_frequency+1) )
        
        inverse_document_frequency_dict[col_name] = log_inverse_document_frequency
        
        tf_idf_df[col_name] = feature_df[col_name] * log_inverse_document_frequency
    
    tf_idf_df['comic_id'] = histogram_df['comic_id']  
    tf_idf_df = tf_idf_df[ ['comic_id']+feature_cols_lst]  
    return (feature_df, inverse_document_frequency_dict, tf_idf_df)



if __name__ == '__main__':
    
    cld_filepath = r"C:\Users\Suraj Shashidhar\Documents\thesis\output\cld_reduced_dim_df.csv" #r"C:\Users\Suraj Shashidhar\Documents\thesis\output\cld_reduced_dim_df_20220917.xlsx"
    edh_filepath = r"C:\Users\Suraj Shashidhar\Documents\thesis\output\edh_reduced_dim_df.csv" # r"C:\Users\Suraj Shashidhar\Documents\thesis\output\edh_reduced_dim_df_20220917.xlsx"
    hog_filepath = r"C:\Users\Suraj Shashidhar\Documents\thesis\output\hog_reduced_dim_df.csv" # r"C:\Users\Suraj Shashidhar\Documents\thesis\output\hog_reduced_dim_df.csv"
    comic_metadata_filepath = r"C:\Users\Suraj Shashidhar\Documents\thesis\output\comic_metadata_df.csv"
    output_folder_path = r"C:\Users\Suraj Shashidhar\Documents\thesis\output" # r"C:\Users\Suraj Shashidhar\Documents\thesis\output"
    parent_dir = r"C:\Users\Suraj Shashidhar\Documents\thesis\panel_img"
    
    cld_np = read_descriptor_data(cld_filepath, filetype='csv').to_numpy()
    edh_np = read_descriptor_data(edh_filepath, filetype='csv').to_numpy()
    hog_np = read_descriptor_data(hog_filepath, filetype='csv').to_numpy()
    # book_metadata_lst = extract_book_metadata(comic_metadata_filepath, filetype='xlsx')
    book_metadata_lst = extract_book_metadata_from_folderpaths(parent_dir=parent_dir, image_format='jpg')
    print(book_metadata_lst[:10])
    print(cld_np.shape)
    
    if os.path.exists(os.path.join(output_folder_path, "kmeans_cld.pickle")):
        with open(os.path.join(output_folder_path, "kmeans_cld.pickle"), 'rb') as fp:
            cld_kmeans = pickle.load(fp)

    if os.path.exists(os.path.join(output_folder_path, "kmeans_edh.pickle")):
        with open(os.path.join(output_folder_path, "kmeans_edh.pickle"), 'rb') as fp:
            edh_kmeans = pickle.load(fp)

    if os.path.exists(os.path.join(output_folder_path, "kmeans_hog.pickle")):
        with open(os.path.join(output_folder_path, "kmeans_hog.pickle"), 'rb') as fp:
            hog_kmeans = pickle.load(fp)
    
    
    # extract visual words, 128 words for each and build histogram
    build_visual_words(descriptor_np=cld_np, book_metadata_lst=book_metadata_lst, num_visual_words=128, output_folder_path=output_folder_path,descriptor_type='cld', model=cld_kmeans )
    build_visual_words(descriptor_np=edh_np, book_metadata_lst=book_metadata_lst, num_visual_words=128, output_folder_path=output_folder_path,descriptor_type='edh', model=edh_kmeans )
    build_visual_words(descriptor_np=hog_np, book_metadata_lst=book_metadata_lst, num_visual_words=128, output_folder_path=output_folder_path,descriptor_type='hog', model=hog_kmeans )
    
    # build TF-IDF vector for each book
    cld_codebook_df = pd.read_csv(os.path.join(output_folder_path,"cld_codebook_df.csv"))
    edh_codebook_df = pd.read_csv(os.path.join(output_folder_path,"edh_codebook_df.csv"))
    hog_codebook_df = pd.read_csv(os.path.join(output_folder_path,"hog_codebook_df.csv"))
    
    cld_feature_df, cld_inverse_document_frequency_dict, cld_tf_idf_df = find_tf_idf_of_bovw(cld_codebook_df)
    cld_tf_idf_df.to_csv(os.path.join(output_folder_path,'cld_tf_idf_df.csv'), index=False)
    
    edh_feature_df, edh_inverse_document_frequency_dict, edh_tf_idf_df = find_tf_idf_of_bovw(edh_codebook_df)
    edh_tf_idf_df.to_csv(os.path.join(output_folder_path,'edh_tf_idf_df.csv'), index=False)
    
    hog_feature_df, hog_inverse_document_frequency_dict, hog_tf_idf_df = find_tf_idf_of_bovw(hog_codebook_df)
    hog_tf_idf_df.to_csv(os.path.join(output_folder_path,'hog_tf_idf_df.csv'), index=False)
    
    
    
    
    
    


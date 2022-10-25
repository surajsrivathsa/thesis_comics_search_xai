import os, sys
from tkinter import image_types
import pandas as pd, numpy as np
import re, shutil, glob, cv2
import pickle
from sklearn.decomposition import PCA
import time

# from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool

# descriptor import
import ColorLayoutComputer as CLD, EdgeHistogramComputer as EDH, OrientedGradientsComputer as HOG

# import skimage.feature._hog as HOG

cld_obj = CLD.ColorLayoutComputer()
edh_obj = EDH.EdgeHistogramComputer(8, 8)
hog_obj = HOG.OrientedGradientsComputer(8, 8, 0.2)

cld_pca = PCA(n_components=64, random_state=5)
edh_pca = PCA(n_components=128, random_state=5)
hog_pca = PCA(n_components=128, random_state=5)
output_folder_path = r"C:\Users\Suraj Shashidhar\Documents\thesis\famous_comics_titles\output"

if os.path.exists(os.path.join(output_folder_path, "cld_pca.pickle")):
    with open(os.path.join(output_folder_path, "cld_pca.pickle"), 'rb') as fp:
        cld_pca = pickle.load(fp)

if os.path.exists(os.path.join(output_folder_path, "edh_pca.pickle")):
    with open(os.path.join(output_folder_path, "edh_pca.pickle"), 'rb') as fp:
        edh_pca = pickle.load(fp)

if os.path.exists(os.path.join(output_folder_path, "hog_pca.pickle")):
    with open(os.path.join(output_folder_path, "hog_pca.pickle"), 'rb') as fp:
        hog_pca = pickle.load(fp)



def save_to_separate_dataframes(descriptor_results_lst: list, output_folder_path: str):
    comic_metadata_df = pd.DataFrame()
    comic_metadata_array = []

    cld_df = pd.DataFrame()
    cld_descriptor_array = []

    edh_df = pd.DataFrame()
    edh_descriptor_array = []

    hog_df = pd.DataFrame()
    hog_descriptor_array = []

    for idx, row in enumerate(descriptor_results_lst):
        # {'comic_id': img_comicid, 'page_num': img_pagenum, 'panel_num': img_panelnum, 'image_filepath': img_filepath, 'cld_descriptor': cld_descriptor, 'edh_descriptor': edh_descriptor, 'hog_descriptor':hog_descriptor}

        comic_metadata_array.append(
            {
                "comic_id": row["comic_id"],
                "page_num": row["page_num"],
                "panel_num": row["panel_num"],
                "image_filepath": row["image_filepath"],
            }
        )
        cld_descriptor_array.append(row["cld_descriptor"])
        edh_descriptor_array.append(row["edh_descriptor"])
        hog_descriptor_array.append(row["hog_descriptor"])

    # deleting to free up memory
    del descriptor_results_lst

    
    comic_metadata_df = pd.DataFrame.from_records(comic_metadata_array)
    print(comic_metadata_df.shape)
    comic_metadata_df.to_excel(
        os.path.join(output_folder_path, "comic_metadata_df.xlsx"), index=False
    )

    print()
    print(" ======= ======== ======= ")
    print()

    cld_df = pd.DataFrame(np.array(cld_descriptor_array))
    print(cld_df.shape)
    cld_df.to_excel(os.path.join(output_folder_path, "cld_df.xlsx"), index=False)
    
    # cld_pca.fit(cld_df)
    cld_reduced_dim_df = pd.DataFrame(cld_pca.transform(cld_df))
    
    print(
        "cld pca: {}".format(
            sum(cld_pca.explained_variance_ratio_)
        )
    )
    cld_reduced_dim_df.to_excel(
        os.path.join(output_folder_path, "cld_reduced_dim_df.xlsx"), index=False
    )
    with open(os.path.join(output_folder_path, "cld_pca.pickle"), "wb") as handle:
        pickle.dump(cld_pca, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del cld_df, cld_reduced_dim_df
    print()
    print(" ======= ======== ======= ")
    print()

    edh_df = pd.DataFrame(np.array(edh_descriptor_array))
    print(edh_df.shape)
    edh_df.to_excel(os.path.join(output_folder_path, "edh_df.xlsx"), index=False)
    
    # edh_pca.fit(edh_df)
    edh_reduced_dim_df = pd.DataFrame(edh_pca.transform(edh_df))
    print(
        "edh pca: {}".format(
            sum(edh_pca.explained_variance_ratio_)
        )
    )
    edh_reduced_dim_df.to_excel(
        os.path.join(output_folder_path, "edh_reduced_dim_df.xlsx"), index=False
    )
    with open(os.path.join(output_folder_path, "edh_pca.pickle"), "wb") as handle:
        pickle.dump(edh_pca, handle, protocol=pickle.HIGHEST_PROTOCOL)

    del edh_df, edh_reduced_dim_df
    print()
    print(" ======= ======== ======= ")
    print()
    
    
    hog_df = pd.DataFrame(np.array(hog_descriptor_array))
    print(hog_df.shape)
    hog_df.to_csv(
        os.path.join(output_folder_path, "hog_df.csv"), index=False
        )
    
    # hog_pca.fit(hog_df)
    hog_reduced_dim_df = pd.DataFrame(hog_pca.transform(hog_df))
    print(
        "hog pca: {}".format(
            sum(hog_pca.explained_variance_ratio_)
        )
    )
    hog_reduced_dim_df.to_csv(
        os.path.join(output_folder_path, "hog_reduced_dim_df.csv"), index=False
    )
    with open(os.path.join(output_folder_path, "hog_pca.pickle"), "wb") as handle:
        pickle.dump(hog_pca, handle, protocol=pickle.HIGHEST_PROTOCOL)

    del hog_df, hog_reduced_dim_df
    print()
    print(" ======= ======== ======= ")
    print()
    return


def bulk_extractor(
    parent_dir: str,
    image_format: str,
    image_descriptor_dict: dict,
    output_folder_path: str,
    num_prcs: int,
    chunksize: int,
    save_to_pickle=True,
    save_to_dataframe=True,
    allowable_subfolder_lst = []
):
    prcs_start_time = time.time()
    
    img_filepath_lst = []
    img_comicid_lst = []
    img_pagenum_lst = []
    img_panelnum_lst = []

    # '/path/**/*.c'
    for f in glob.glob(
        os.path.join(parent_dir, "**", "*." + image_format), recursive=True
    ):
        if allowable_subfolder_lst:
            if os.path.basename(os.path.dirname(f)) in allowable_subfolder_lst:
                img_filepath_lst.append(f)

                img_comicid_lst.append(os.path.basename(os.path.dirname(f)))

                bsname = os.path.basename(f).replace("." + image_format, "").split("_")

                img_pagenum_lst.append(bsname[1])
                img_panelnum_lst.append(bsname[-1]) 
        else:
            img_filepath_lst.append(f)

            img_comicid_lst.append(os.path.basename(os.path.dirname(f)))

            bsname = os.path.basename(f).replace("." + image_format, "").split("_")

            img_pagenum_lst.append(bsname[1])
            img_panelnum_lst.append(bsname[-1])

    print('number of images: {}'.format(len(img_filepath_lst)))
    pool = Pool(num_prcs)

    descriptor_results_lst = pool.starmap(
        single_image_extractor,
        zip(img_filepath_lst, img_comicid_lst, img_pagenum_lst, img_panelnum_lst),
        chunksize=chunksize,
    )
    
    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()

    print('processing completed: {}'.format( ( time.time()-prcs_start_time )/60.0 ))
    print(time.time())
    
    if save_to_pickle:
        with open(
            os.path.join(output_folder_path, "descriptors.pickle"), "wb"
        ) as handle:
            pickle.dump(
                descriptor_results_lst, handle, protocol=pickle.HIGHEST_PROTOCOL
            )

    if save_to_dataframe:
        save_to_separate_dataframes(descriptor_results_lst, output_folder_path)

    return


def single_image_extractor(
    img_filepath: str, img_comicid="", img_pagenum="", img_panelnum=""
):

    img = cv2.imread(img_filepath)
    cld_descriptor = cld_obj.compute(img)
    edh_descriptor = edh_obj.compute(img).ravel()
    hog_descriptor = hog_obj.compute(img).ravel()

    return {
        "comic_id": img_comicid,
        "page_num": img_pagenum,
        "panel_num": img_panelnum,
        "image_filepath": img_filepath,
        "cld_descriptor": cld_descriptor,
        "edh_descriptor": edh_descriptor,
        "hog_descriptor": hog_descriptor,
    }


if __name__ == "__main__":
    start_time = time.time()
    print(start_time)
    # required_folders_df = pd.read_csv(r"C:\Users\Suraj Shashidhar\Documents\thesis\comicnum_to_book_title - Sheet1.csv").astype(str)
    # testdata_subfolder_lst = list(required_folders_df['comic_no'].unique()) # ['3451', '3452', '3453', '3454', '3455', '3460', '3461', '3462', '3463', '3464', '3465', '3466', '3477'] #list(required_folders_df['comic_no'].unique())
    output_folder_path = r"C:\Users\Suraj Shashidhar\Documents\thesis\famous_comics_titles\output"
    parent_dir = r"C:\Users\Suraj Shashidhar\Documents\thesis\famous_comics_titles\panel_img" # r"C:\Users\Suraj Shashidhar\Documents\thesis\raw_panel_images\raw_panel_images"
    image_format = "jpg"
    image_descriptor_dict = {}
    # output_folder_path = "/Users/surajshashidhar/git/thesis_comics_search_xai/data"
    num_prcs = 10
    chunksize = 50

    bulk_extractor(
        parent_dir,
        image_format,
        image_descriptor_dict,
        output_folder_path,
        num_prcs,
        chunksize,
        allowable_subfolder_lst=[]
    )

    print()
    print("Time Taken in minutes: {}".format((time.time() - start_time) / 60.0))


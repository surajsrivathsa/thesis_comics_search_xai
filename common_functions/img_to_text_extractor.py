import os, sys, glob, pathlib, re
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from PIL import Image
from PIL import Image, ImageEnhance, ImageFilter
import time
import statistics
import traceback

from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool as ProcessPool
from multiprocessing import cpu_count
import multiprocessing

from autocorrect import Speller
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Suraj Shashidhar\AppData\Local\Tesseract-OCR\tesseract.exe"
spell = Speller()

def convert_img_page_to_text(page_path: str):
    try:
        im = Image.open(page_path) # the second one 
        im = im.filter(ImageFilter.MedianFilter())
        enhancer = ImageEnhance.Contrast(im)
        im = enhancer.enhance(2)
        im = im.convert('1')
        text = pytesseract.image_to_string(im)
    except Exception as e:
        text = ''
        print('Cannot process page: {}'.format(page_path), flush=True)
    finally:
        return text
    

def process_text_one_page(img_filepath: str, individual_comic_name: str, comic_id, page_idx: int, our_idx: int):    
    try:
        text = convert_img_page_to_text(img_filepath)
        cleaned_text = spell(text)
    except Exception as e:
        cleaned_text = ''
        
    record = { 'comic_no': comic_id, 'book_title': individual_comic_name, 'page_no': page_idx, 'text': cleaned_text, 'our_idx': our_idx }
    
    return record


def process_comic_book_text(comic_info_dict: dict):
    
    
    comic_id = comic_info_dict['comic_id']
    individual_comic_name = comic_info_dict['individual_comic_name']
    parent_folder_path = comic_info_dict['parent_folder_path']
    our_idx = comic_info_dict['our_idx']
    full_img_folder_path = os.path.join(parent_folder_path, individual_comic_name)
    
    img_filepath_lst =  glob.glob( os.path.join(full_img_folder_path, "*.jpeg")  )
    individual_comic_name_lst = [individual_comic_name for x in range(len(img_filepath_lst))]
    comic_id_lst = [comic_id for x in range(len(img_filepath_lst))]
    page_idx_lst = [x for x in range(len(img_filepath_lst))]
    our_idx_lst = [our_idx for x in range(len(img_filepath_lst))]
    full_book_text_record_lst= []
    
    """
    num_threads = 4
    threadpool = ThreadPool(num_threads)

    full_book_text_record_lst = threadpool.starmap( process_text_one_page, zip(img_filepath_lst, individual_comic_name_lst, comic_id_lst, page_idx_lst, our_idx_lst), chunksize=4)
    threadpool.close()
    threadpool.join()
    """
    
    
    for idx, (img_filepath, individual_comic_name, comic_id, page_idx, our_idx) in enumerate(zip(img_filepath_lst,individual_comic_name_lst, comic_id_lst, page_idx_lst, our_idx_lst )):
        page_record = process_text_one_page(img_filepath, individual_comic_name, comic_id, page_idx, our_idx)
        full_book_text_record_lst.append(page_record)
    
    print('processed: {}'.format(individual_comic_name), flush=True)
    print() 
    return full_book_text_record_lst
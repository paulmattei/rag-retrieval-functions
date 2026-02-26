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
        display_widget
    )

NEWS_DATA = read_dataframe("news_data_dedup.csv")

pprint(NEWS_DATA)
              

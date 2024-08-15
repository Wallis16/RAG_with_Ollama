from datasets import DatasetDict
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

import os

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF")

ST = SentenceTransformer("all-mpnet-base-v2")

def embed(dataset):
    return {"embeddings" : ST.encode(dataset["text"])}

def create_embeddings(dataset, batch_size):

    dataset_100_rows_with_embbedings = dataset.map(embed,batched=True,batch_size=batch_size)
    return dataset_100_rows_with_embbedings

def push_to_huggingface(dataset, hub_path, revision):

    dataset_dict = DatasetDict({
        'train': dataset
    })
    dataset_dict.push_to_hub(hub_path, revision=revision)
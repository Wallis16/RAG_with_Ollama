from datasets import load_dataset, DatasetDict
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

import os

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF")

def embed(dataset):
    return {"embeddings" : ST.encode(dataset["text"])}

dataset_100_rows = load_dataset("wikipedia", "20220301.simple",trust_remote_code=True, split='train[:100]')

ST = SentenceTransformer("all-mpnet-base-v2")

dataset_100_rows_with_embbedings = dataset_100_rows.map(embed,batched=True,batch_size=16)

# Wrap in DatasetDict if needed
dataset_100_rows_dict = DatasetDict({
    'train': dataset_100_rows
})

# Push the dataset with embeddings to the Hugging Face Hub
dataset_100_rows_dict.push_to_hub("diogenes-wallis/20220301_simple_100_all_mpnet_base_v2")

# Wrap in DatasetDict if needed
dataset_100_rows_with_embbedings_dict = DatasetDict({
    'train': dataset_100_rows_with_embbedings
})

# Push the dataset with embeddings to the Hugging Face Hub
dataset_100_rows_with_embbedings_dict.push_to_hub("diogenes-wallis/20220301_simple_100_all_mpnet_base_v2", revision="embedded")
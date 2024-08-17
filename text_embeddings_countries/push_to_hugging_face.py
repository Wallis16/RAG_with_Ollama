import os
from dotenv import load_dotenv
from datasets import Dataset, DatasetDict
import pandas as pd

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF")

df = pd.read_csv('countries_dataset.csv')

dataset = Dataset.from_pandas(df)
dataset_dict = DatasetDict({
    'train': dataset
})
dataset_dict.push_to_hub("diogenes-wallis/wikipedia-all-countries", revision="main")
print("Dataset uploaded successfully to Hugging Face!")
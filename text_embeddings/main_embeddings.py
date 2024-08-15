from datasets import load_dataset
from generate_embeddings import create_embeddings, push_to_huggingface

path = "wikipedia"
name = "20220301.simple"
hub_path = "diogenes-wallis/20220301_simple_100_all_mpnet_base_v2"
batch_size = 16
dataset = load_dataset(path, name,trust_remote_code=True, split='train[:120]')

push_to_huggingface(dataset, hub_path, revision="main")

embeddings = create_embeddings(dataset, batch_size)
push_to_huggingface(embeddings, hub_path, revision="embedded")
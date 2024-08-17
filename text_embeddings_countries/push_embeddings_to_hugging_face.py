from datasets import load_dataset
from text_embeddings.generate_embeddings import create_embeddings, push_to_huggingface

hub_path = "diogenes-wallis/wikipedia-all-countries"
batch_size = 16

dataset = load_dataset(hub_path, trust_remote_code=True, split='train[:]')
embeddings = create_embeddings(dataset, batch_size)
push_to_huggingface(embeddings, hub_path, revision="embedded")
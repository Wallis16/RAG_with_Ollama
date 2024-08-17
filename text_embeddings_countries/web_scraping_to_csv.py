#code created with chatgpt

import wikipediaapi
import pandas as pd
import pycountry

# Initialize the Wikipedia API
wiki_wiki = wikipediaapi.Wikipedia('RAG for countries (email)', 'en')

# Fetch the list of all countries
countries = [country.name for country in pycountry.countries]

# Initialize the list to store data
data = []

for country in countries:
    try:
        # Fetch the Wikipedia page for the country
        page = wiki_wiki.page(country)
        
        # Extract required details
        page_id = page.pageid
        page_url = page.fullurl
        page_title = page.title
        page_text = page.text

        # Append the data to the list
        data.append({
            'id': page_id,
            'url': page_url,
            'title': page_title,
            'text': page_text
        })
    except Exception as e:
        print(f"Error processing page for {country}: {e}")

# Create a DataFrame from the data
df = pd.DataFrame(data, columns=['id', 'url', 'title', 'text'])

# Save DataFrame to a CSV file
df.to_csv('countries_dataset.csv', index=False)

print("Dataset created and saved as 'countries_dataset.csv'")

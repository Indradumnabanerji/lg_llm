from langchain.llms import HuggingFaceHub
import concurrent.futures
from collections import Counter
import string
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from os import getenv
import matplotlib.pyplot as plt

HF_TOKEN="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"
access_token = getenv(HF_TOKEN)

def get_gpt_response(prompt):
    mistral = HuggingFaceHub(repo_id="mistralai/Mistral-7B-v0.1", huggingfacehub_api_token=HF_TOKEN)
    return mistral(prompt)

# Function to create a word cloud from responses
def create_word_cloud(responses):
    all_words = ' '.join(responses).lower()
    all_words = all_words.translate(str.maketrans('', '', string.punctuation))

    # Create and display the word cloud
    wordcloud = WordCloud(width=800, height=400).generate(all_words)
    wordcloud.to_file("word_cloud.png")  # Save the word cloud as an image file
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

# Function to process responses and combine them
def process_responses(responses):
    all_words = ' '.join(responses).lower()
    all_words = all_words.translate(str.maketrans('', '', string.punctuation))
    word_counts = Counter(all_words.split())

    common_words = {word for word, count in word_counts.items() if count > 1}
    combined_response = []

    for response in responses:
        words = response.split()
        unique_words = [word for word in words if word.lower() not in common_words]
        combined_response.append(' '.join(unique_words))

    return ' '.join(combined_response)

# Function to calculate similarity score between a new response and the existing word cloud
def calculate_similarity(new_response, word_cloud_responses):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(word_cloud_responses + [new_response])

    # Calculate cosine similarity between the new response and the existing word cloud responses
    similarity_scores = cosine_similarity(tfidf_matrix[:-1], tfidf_matrix[-1:])
    
    return similarity_scores[0][0]

# Main execution
prompts = ["Write a poem about a robot falling in love with a human."]

responses = []

with concurrent.futures.ThreadPoolExecutor() as executor:
    for prompt in prompts:
        future_responses = [executor.submit(get_gpt_response, prompt) for _ in range(3)]
        results = [future.result() for future in concurrent.futures.as_completed(future_responses)]
        processed_response = process_responses(results)
        print(f"\n\nCombined HF-Mistral LLM Response ({prompt}):", processed_response)
        responses.append(processed_response)

# Create a word cloud from the responses
create_word_cloud(responses)

#new_response =[]
# Test a new response for similarity

prompts = ["Write a poem about a robot falling in love with a human."]

mistral = HuggingFaceHub(repo_id="mistralai/Mistral-7B-v0.1", huggingfacehub_api_token=HF_TOKEN)

for prompt in prompts:
    print(f"\n\nHF-Mistral LLM ({prompt}):", mistral(prompt))

similarity_score = calculate_similarity(mistral(prompt), responses)
print("\nSimilarity Score with Word Cloud Responses:", similarity_score)

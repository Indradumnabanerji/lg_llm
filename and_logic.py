from langchain.llms import HuggingFaceHub
import concurrent.futures
from collections import Counter
import string
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from os import getenv
import matplotlib.pyplot as plt

HF_TOKEN = "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"
access_token = getenv(HF_TOKEN)

def get_gpt_response(prompt):
    mistral = HuggingFaceHub(repo_id="mistralai/Mistral-7B-v0.1", huggingfacehub_api_token=HF_TOKEN)
    return mistral(prompt)

# Modified function to find common words across responses
def find_common_words(responses):
    # Tokenizing and cleaning the responses
    tokenized_responses = [set(response.lower().translate(str.maketrans('', '', string.punctuation)).split()) for response in responses]
    # Finding common words
    common_words = set.intersection(*tokenized_responses)
    return ' '.join(common_words)

# Function to create a word cloud
def create_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400).generate(text)
    wordcloud.to_file("word_cloud.png")  # Save as an image file
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

# Function to calculate similarity score
def calculate_similarity(new_response, common_words_text):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([common_words_text, new_response])

    # Calculating cosine similarity
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

    return similarity_scores[0][0]

# Main execution
prompts = ["Write a poem about a robot falling in love with a human."]
responses = []

with concurrent.futures.ThreadPoolExecutor() as executor:
    future_responses = [executor.submit(get_gpt_response, prompt) for prompt in prompts for _ in range(3)]
    results = [future.result() for future in concurrent.futures.as_completed(future_responses)]

    common_words_text = find_common_words(results)
    print(f"\nCommon Words Across Responses: {common_words_text}")

    # Create a word cloud from common words
    create_word_cloud(common_words_text)

    # Test a new response for similarity
    mistral = HuggingFaceHub(repo_id="mistralai/Mistral-7B-v0.1", huggingfacehub_api_token=HF_TOKEN)
    for prompt in prompts:
        new_response = mistral(prompt)
        print(f"\n\nHF-Mistral LLM ({prompt}):", new_response)

        similarity_score = calculate_similarity(new_response, common_words_text)
        print("\nSimilarity Score with Common Words:", similarity_score)

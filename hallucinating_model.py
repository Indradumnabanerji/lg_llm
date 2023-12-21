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
# ... [Previous imports and function definitions for OR Logic]
def create_word_cloud(responses):
    all_words = ' '.join(responses).lower()
    all_words = all_words.translate(str.maketrans('', '', string.punctuation))

    # Create and display the word cloud
    wordcloud = WordCloud(width=800, height=400).generate(all_words)
    wordcloud.to_file("word_cloud.png")  # Save the word cloud as an image file
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

def get_hallucinatory_response_or(prompt):
    modified_prompt = hallucinatory_prompt(prompt)
    response = mistral(modified_prompt)
    return post_process_to_hallucinate(response)

# Main execution for OR Logic
prompts = ["Write a poem about a robot falling in love with a human."]
or_responses = []

with concurrent.futures.ThreadPoolExecutor() as executor:
    future_responses = [executor.submit(get_hallucinatory_response_or, prompt) for prompt in prompts for _ in range(3)]
    results = [future.result() for future in concurrent.futures.as_completed(future_responses)]
    processed_response = process_responses(results)
    print(f"\n\nOR Logic - Combined Hallucinatory Response ({prompt}):", processed_response)
    or_responses.append(processed_response)

# Create a word cloud from the OR logic responses
create_word_cloud(or_responses)

# Evaluate similarity for OR logic
for prompt in prompts:
    new_response = mistral(prompt)
    print(f"\n\nHF-Mistral LLM ({prompt}):", new_response)
    similarity_score_or = calculate_similarity(new_response, or_responses)
    print("\nOR Logic - Similarity Score with Word Cloud Responses:", similarity_score_or)

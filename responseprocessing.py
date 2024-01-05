# Import necessary libraries
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import string
import matplotlib.pyplot as plt
from collections import Counter

class ResponseProcessor:
    def __init__(self, hf_token, cerebras_model_dir, openai_api_key):
        self.hf_token = hf_token
        self.cerebras_model_dir = cerebras_model_dir
        self.openai_api_key = openai_api_key
        
    # Function to create a word cloud from responses
    def create_word_cloud(self, responses):
        all_words = ' '.join(responses).lower()
        all_words = all_words.translate(str.maketrans('', '', string.punctuation))

        # Create and display the word cloud
        wordcloud = WordCloud(width=800, height=400).generate(all_words)
        wordcloud.to_file("word_cloud.png")  # Save the word cloud as an image file
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()

    # Function to process responses and combine them
    def process_responses(self, responses):
        # Combine all words from all responses
        combined_response = ' '.join(responses).lower()
        # Remove punctuation
        combined_response = combined_response.translate(str.maketrans('', '', string.punctuation))
        return combined_response

    # Function to calculate similarity score between a new response and the existing word cloud
    def get_sentence_embeddings(self, sentences):
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(sentences)
        return embeddings

    def calculate_semantic_similarity(self, new_response, word_cloud_responses):
        responses = word_cloud_responses + [new_response]
        embeddings = self.get_sentence_embeddings(responses)
        similarity_matrix = cosine_similarity(embeddings)

        return similarity_matrix[-1, :-1]
    
    # Function to find common words across responses
    def find_common_words(self, responses):
        # Tokenizing and cleaning the single string of responses
        word_counts = {}
        words = str(responses).split()
        word_counts.update(Counter(words))
        words_common =[]
        for word, count in word_counts.items():
            if count > 3:
                words_common.append(word)
        print(words_common)
        
        return words_common

    
    # Function to process responses and combine them

    def process_hallucinatory_responses(self, responses, prompt, new_prompt):
        # Remove punctuation from prompts and split into words
        prompt_words = set(prompt.lower().translate(str.maketrans('', '', string.punctuation)).split())
        new_prompt_words = set(new_prompt.lower().translate(str.maketrans('', '', string.punctuation)).split())

        # Combine all words from all responses
        combined_response = ' '.join(responses).lower()

        # Remove punctuation from combined responses
        combined_response = combined_response.translate(str.maketrans('', '', string.punctuation))

        # Split combined response into words
        response_words = combined_response.split()

        # Count the frequency of each word in the combined response
        word_frequency = Counter(response_words)

        # Identify common words and words from new prompt
        common_threshold = 3
        common_words = {word for word, count in word_frequency.items() if count > common_threshold or word in new_prompt_words}

        # Remove words that are in the original prompt, new prompt, or are common
        words_filtered = [word for word in response_words if word not in prompt_words and word not in common_words]

        return ' '.join(words_filtered)


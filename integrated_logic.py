# Import necessary libraries
import concurrent.futures
from goodresponse import GoodResponse
from hallucinatingresponse import HallucinatedResponse
from responseprocessing import ResponseProcessor

# Set up API keys and model directories
HF_TOKEN = "your_hf_token"
OPENAI_API_KEY = 'your_open_api_key'
CEREBRAS_MODEL_DIR = r"C:\Users\indra\OneDrive\Desktop\GitHub\lg_llm\Checkpoints"

# Instantiate classes
response_generator = GoodResponse(HF_TOKEN, CEREBRAS_MODEL_DIR, OPENAI_API_KEY)
hallucinator_generator = HallucinatedResponse(HF_TOKEN, CEREBRAS_MODEL_DIR, OPENAI_API_KEY)
response_processor = ResponseProcessor(HF_TOKEN, CEREBRAS_MODEL_DIR, OPENAI_API_KEY)

# Define prompts
prompts = ["When are the holidays?"]
newprompt = "When are the holidays?"

# Main execution
responses = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    # OR Logic
    for prompt in prompts:
        future_responses = [executor.submit(response_generator.get_gpt2_response, prompt) for _ in range(5)]
        results = [future.result() for future in concurrent.futures.as_completed(future_responses)]
        combined_response = response_processor.process_responses(results)
        responses.append(combined_response)

    # AND Logic
    future_responses_and = []
    for prompt in prompts:
        future_responses_and.extend([executor.submit(response_generator.get_mistral_response, prompt) for _ in range(5)])
    common_words_text = response_processor.find_common_words([future.result() for future in concurrent.futures.as_completed(future_responses_and)])
    responses.append(common_words_text)

    # NOT OR Logic
    future_responses_not_or = []
    for prompt in prompts:
        future_responses_not_or.extend([executor.submit(hallucinator_generator.get_mistral_response, prompt) for _ in range(5)])
    hallucinated_responses = response_processor.process_hallucinatory_responses([future.result() for future in concurrent.futures.as_completed(future_responses_not_or)], prompt, newprompt)
    responses.append(hallucinated_responses)

# Generate a new response and calculate similarity scores
new_response = response_generator.get_mistral_response(newprompt)
print("New Response:", new_response)
for response_set in responses:
    similarity_score = response_processor.calculate_semantic_similarity(new_response, [response_set])
    print("\nSimilarity Score with Word Cloud Responses:", similarity_score)

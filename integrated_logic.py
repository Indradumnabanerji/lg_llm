# Import necessary libraries
import concurrent.futures
from goodresponse import GoodResponse
from hallucinatingresponse import HallucinatedResponse
from responseprocessing import ResponseProcessor
import csv
import os

k = 0
for k in range(3): 
    # Set up API keys and model directories
    HF_TOKEN = ""
    OPENAI_API_KEY = ''
    CEREBRAS_MODEL_DIR = r"C:\Users\indra\OneDrive\Desktop\GitHub\lg_llm\Checkpoints"

    # Instantiate classes
    response_generator = GoodResponse(HF_TOKEN, CEREBRAS_MODEL_DIR, OPENAI_API_KEY)
    hallucinator_generator = HallucinatedResponse(HF_TOKEN, CEREBRAS_MODEL_DIR, OPENAI_API_KEY)
    response_processor = ResponseProcessor(HF_TOKEN, CEREBRAS_MODEL_DIR, OPENAI_API_KEY)

    # Define prompts
    prompt = "astronauts playing football"
    newprompt = "astronauts playing football"

    # Main execution
    responses_or = []
    responses_and = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # OR Logic
        future_mistral_responses = [executor.submit(response_generator.get_mistral_response, prompt) for _ in range(3)]
        future_gpt2_responses = [executor.submit(response_generator.get_gpt2_response, prompt) for _ in range(3)]
        future_gpt3_responses = [executor.submit(response_generator.get_gpt3_response, prompt) for _ in range(3)]
        future_cerebras_responses = [executor.submit(response_generator.get_cerebras_response, prompt) for _ in range(3)]

        all_futures = future_mistral_responses + future_gpt2_responses + future_gpt3_responses + future_cerebras_responses
        results = [future.result() for future in concurrent.futures.as_completed(all_futures)]

        # Process responses for OR Logic
        combined_response_or = response_processor.process_responses(results)
        responses_or.append(combined_response_or)

        # AND Logic - Use find_common_words function
        common_words_text = response_processor.find_common_words(results)
        print (common_words_text)
        combined_response_and = response_processor.create_word_cloud(common_words_text)  # Assuming this function processes and returns a response
        responses_and.append(common_words_text)


    # Print OR and AND logic responses
    print("OR Logic Responses:", responses_or)
    print("AND Logic Responses:", responses_and)

    # Hallucination Logic
    # NOT XOR Score
    responses_not_xor = []
    responses_not_and = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # OR Logic
        future_mistral_not_responses = [executor.submit(hallucinator_generator.get_mistral_response, prompt) for _ in range(3)]
        future_gpt2_not_responses = [executor.submit(hallucinator_generator.get_gpt2_response, prompt) for _ in range(3)]
        future_gpt3_not_responses = [executor.submit(hallucinator_generator.get_gpt3_response, prompt) for _ in range(3)]
        future_cerebras_not_responses = [executor.submit(hallucinator_generator.get_cerebras_response, prompt) for _ in range(3)]
        
        all_not_futures = future_mistral_not_responses + future_gpt2_not_responses + future_gpt3_not_responses + future_cerebras_not_responses
        results_not = [future.result() for future in concurrent.futures.as_completed(all_not_futures)]

        # Process responses for OR Logic
        combined_response_not_xor = response_processor.process_hallucinatory_responses(results, prompt, newprompt)
        responses_not_xor.append(combined_response_not_xor)
        

        # AND Logic - Use find_common_words function
        common_not_text = response_processor.find_common_words(results_not)
        print (common_not_text)
        combined_response_not_and = response_processor.create_word_cloud(common_not_text)  # Assuming this function processes and returns a response
        responses_not_and.append(common_not_text)

    # Generate a new response and calculate similarity scores

    new_response = hallucinator_generator.get_mistral_response(newprompt)
    print("New Response:", new_response)
    Or_similarity_score = response_processor.calculate_semantic_similarity(new_response, responses_or)

    And_similarity_score = response_processor.calculate_semantic_similarity(new_response, responses_and)

    # Generate a new response and calculate similarity scores

    Not_xor_similarity_score = response_processor.calculate_semantic_similarity(new_response, responses_not_xor)
    Not_and_similarity_score = response_processor.calculate_semantic_similarity(new_response, responses_not_and)

    print("\n Or Similarity Score with Word Cloud Responses:", Or_similarity_score[0])
    print("\n And Similarity Score with Word Cloud Responses:", And_similarity_score[0])
    print("\n Not xor similarity score with Word Cloud Responses:", Not_xor_similarity_score[0])
    print("\n Not and similarity score with Word Cloud Responses:", Not_and_similarity_score[0])


    # CSV file name
    csv_file = "Logic_scores.csv"
    # Check if file exists
    file_exists = os.path.exists(csv_file)

    # Writing data to the CSV file
    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:  # mode='a' for appending
        writer = csv.writer(file)

        # Writing the header if file does not exist
        if not file_exists:
            writer.writerow(["Model", "OR Similarity Score", "AND Similarity Score", "NOT XOR Similarity Score", "NOT AND Similarity Score", "Prompt", "New Prompt"])

        # Writing the data
        writer.writerow(["mistral", Or_similarity_score[0], And_similarity_score[0], Not_xor_similarity_score[0], Not_and_similarity_score[0], prompt, newprompt])

    print(f"Data written to {csv_file}")
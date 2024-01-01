# Import necessary libraries
from langchain.llms import HuggingFaceHub
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM, pipeline
from openai import OpenAI


class GoodResponse:
    def __init__(self, hf_token, cerebras_model_dir, openai_api_key):
        self.hf_token = hf_token
        self.cerebras_model_dir = cerebras_model_dir
        self.openai_api_key = openai_api_key

    def get_mistral_response(self, prompt):
        mistral = HuggingFaceHub(repo_id="mistralai/Mistral-7B-v0.1", huggingfacehub_api_token=self.hf_token)
        return mistral(prompt, temperature=0.1, top_p=0.9)

    def get_gpt2_response(self, prompt):
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        inputs = tokenizer.encode(prompt, return_tensors='pt')
        outputs = model.generate(inputs, max_length=500, num_return_sequences=1, temperature=0.1, top_p=0.9, do_sample=True)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def get_gpt3_response(self, prompt):
        client = OpenAI(api_key=self.openai_api_key)
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Your task is to generate creative responses"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            top_p=0.9,
            max_tokens=50
        )
        return completion.choices[0].message.content

    def get_cerebras_response(self, prompt):
        tokenizer = AutoTokenizer.from_pretrained(self.cerebras_model_dir)
        model = AutoModelForCausalLM.from_pretrained(self.cerebras_model_dir)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        output = pipe(prompt, max_length=50, do_sample=True, temperature=0.1, no_repeat_ngram_size=2)
        return output[0].get('generated_text', '').replace(prompt, "")

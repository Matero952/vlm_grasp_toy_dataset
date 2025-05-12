from google import genai
class GeminiExperiment:
    def __init__(self, prompt_func):
        self.prompt_func = prompt_func
        self.client = genai.Client(api_key=)
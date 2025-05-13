import anthropic
import os
import base64
import mimetypes
import re
class ClaudeExperiment:
    def __init__(self, model, prompt):
        self.model = model
        self.prompt = prompt
        self.client = anthropic.Anthropic(api_key=os.getenv('CLAUDE'))
    def process_sample(self, img_path):
        with open(img_path, "rb") as img_file:
            converted_img_data = base64.b64encode(img_file.read()).decode("utf-8")
            mime_type, _ = mimetypes.guess_type(img_path)
        message = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[
            {
            "role": "user",
            "content": [
                    {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": converted_img_data,
                        },
                    },
                    {
                    "type": "text",
                    "text": self.prompt
                    }
                ],
            }
        ],
        )
        # result = message.lower()
        grasp = re.search(r"palmar pinch|power disk|precision disk|power sphere|precision sphere|sphere 3 finger|sphere 4 finger|tripod|inferior pincer|quadpod|lateral|extension type|palmar", message.content[0].text.lower())
        return message.content[0].text.lower(), grasp


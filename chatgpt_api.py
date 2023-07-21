import openai
import numpy as np
import pandas as pd
from config import openai_settings, paths

openai.api_key = openai_settings.OPENAI_API_KEY


class RagChatGPT:
    def __init__(self):
        self.prompt = open(
            paths.PROMPT_DIR.joinpath("chatgpt.txt"), encoding="utf-8"
        ).read()
        self.embedidngs = np.load(paths.DATA_DIR.joinpath("embeddings.npy"))
        self.db = self.get_docs()

    def get_docs(self):
        db = pd.read_csv(paths.DATA_DIR.joinpath("example.csv"))
        return db["차량 비상시 응급조치"].tolist()

    def query_embedding(self, user_message: str) -> np.ndarray:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=[user_message],
        )
        query_embedding = np.array(response["data"][0]["embedding"])
        return query_embedding

    def sementic_search(self, user_message: str) -> str:
        query_embedding = self.query_embedding(user_message)
        dot_product = np.dot(self.embedidngs, query_embedding)
        norm_a = np.linalg.norm(query_embedding)
        norm_b = np.linalg.norm(self.embedidngs, axis=1)
        cos_sim = dot_product / (norm_a * norm_b)
        max_index = np.argmax(cos_sim)
        return self.db[max_index]

    def reply(self, user_message: str) -> str:
        instruction = self.prompt + "\n\n" + self.sementic_search(user_message)
        print(instruction)
        messages = [{"role": "system", "content": instruction}]

        messages.append({"role": "user", "content": f"{user_message}"})

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
        answer = completion.choices[0].message["content"].strip()
        # answer = answer.replace("\n", " ")
        return answer


if __name__ == "__main__":
    chat = RagChatGPT()
    print(chat.reply("엔진과열이 발생했어요"))

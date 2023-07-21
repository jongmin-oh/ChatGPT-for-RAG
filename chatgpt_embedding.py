import pandas as pd
import numpy as np
import openai

from config import openai_settings

df = pd.read_csv("./data/example.csv")
docs = df["차량 비상시 응급조치"].tolist()

openai.api_key = openai_settings.OPENAI_API_KEY

response = openai.Embedding.create(
    model="text-embedding-ada-002",
    input=docs,
)

embeddings = np.array([i["embedding"] for i in response["data"]])
np.save("./data/embeddings.npy", embeddings)

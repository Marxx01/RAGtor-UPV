import openai

client = openai.OpenAI(
    api_key="sk-1C-hWjmHEW05iQjwmr9EnA",
    base_url="https://api.poligpt.upv.es"
)

response = client.chat.completions.create(
    model="poligpt",
    messages = [
        {
            "role": "user",
            "content": "¿Cuantas horas de prácticas son equivalentes a 1 crédito?"
        }
    ]
)
print(response)
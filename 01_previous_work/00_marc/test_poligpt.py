import openai

client = openai.OpenAI(
    api_key="", #Pon la key aqui
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

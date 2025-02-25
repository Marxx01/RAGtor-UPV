import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Seleccionar el mejor dispositivo disponible
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Cargar dataset
ds = load_dataset("9wimu9/wiki_support_docs_en", split="train")

# Cargar modelos y tokenizers
model_name_ca = "Helsinki-NLP/opus-mt-en-ca"  # Modelo inglés → catalán
model_name_es = "Helsinki-NLP/opus-mt-en-es"  # Modelo inglés → español

tokenizer_ca = AutoTokenizer.from_pretrained(model_name_ca)
model_ca = AutoModelForSeq2SeqLM.from_pretrained(model_name_ca).to(device)

tokenizer_es = AutoTokenizer.from_pretrained(model_name_es)
model_es = AutoModelForSeq2SeqLM.from_pretrained(model_name_es).to(device)

# Diccionario para almacenar los resultados
data = {}

# Procesar todo en un solo bucle
for i, row in enumerate(ds):
    # Obtener datos originales en inglés
    question = row["question"]
    context = " ".join(row["support_documents"]) if isinstance(row["support_documents"], list) else row["support_documents"]
    response = row["answer"]

    # Tokenizar y mover a dispositivo
    inputs_ca = tokenizer_ca([question, context, response], return_tensors="pt", truncation=True, padding=True).to(device)
    inputs_es = tokenizer_es([question, context, response], return_tensors="pt", truncation=True, padding=True).to(device)

    # Generar traducciones
    outputs_ca = model_ca.generate(**inputs_ca)
    outputs_es = model_es.generate(**inputs_es)

    translations_ca = [tokenizer_ca.decode(output, skip_special_tokens=True) for output in outputs_ca]
    translations_es = [tokenizer_es.decode(output, skip_special_tokens=True) for output in outputs_es]

    # Guardar en diccionario
    data[i] = {
        "en": {
            "pregunta": question,
            "contexto": context,
            "respuesta": response
        },
        "ca": {
            "pregunta": translations_ca[0],
            "contexto": translations_ca[1],
            "respuesta": translations_ca[2]
        },
        "es": {
            "pregunta": translations_es[0],
            "contexto": translations_es[1],
            "respuesta": translations_es[2]
        }
    }

    # Procesar solo los primeros 5 elementos como prueba
    if i >= 4:
        break

# Guardar resultados en un archivo JSON
with open("00_marc/traducciones.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("Traducción completada y guardada en 'traducciones.json'")
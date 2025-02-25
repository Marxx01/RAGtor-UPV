import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Seleccionar el mejor dispositivo disponible
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Cargar dataset completo
ds = load_dataset("9wimu9/wiki_support_docs_en", split="train")

# Modelos de traducción (solo Helsinki-NLP)
model_name_en_es = "Helsinki-NLP/opus-mt-en-es"  # Inglés → Español
model_name_en_ca = "Helsinki-NLP/opus-mt-en-ca"  # Inglés → Catalán
model_name_es_ca = "Helsinki-NLP/opus-mt-es-ca"  # Español → Catalán

# Cargar modelos y tokenizers
tokenizer_en_es = AutoTokenizer.from_pretrained(model_name_en_es)
model_en_es = AutoModelForSeq2SeqLM.from_pretrained(model_name_en_es).to(device)

tokenizer_en_ca = AutoTokenizer.from_pretrained(model_name_en_ca)
model_en_ca = AutoModelForSeq2SeqLM.from_pretrained(model_name_en_ca).to(device)

tokenizer_es_ca = AutoTokenizer.from_pretrained(model_name_es_ca)
model_es_ca = AutoModelForSeq2SeqLM.from_pretrained(model_name_es_ca).to(device)

data = {}

# Función para dividir textos
def split_text(text, max_length=300):
    sentences = text.split(". ")
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    chunks.append(current_chunk.strip())
    return chunks

# Función para traducir
def translate_text(model, tokenizer, text):
    text_chunks = split_text(text, max_length = 300)
    translated_chunks = []
    
    for chunk in text_chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        outputs = model.generate(**inputs, max_length = 512)
        translated_chunks.append(tokenizer.decode(outputs[0], skip_special_tokens=True))

    return " ".join(translated_chunks)

for i, row in enumerate(ds):
    question = row["question"]
    context = " ".join(row["support_documents"]) if isinstance(row["support_documents"], list) else row["support_documents"]
    response = row["answer"]

    # Traducciones al español
    translations_es = {
        "pregunta": translate_text(model_en_es, tokenizer_en_es, question),
        "contexto": translate_text(model_en_es, tokenizer_en_es, context),
        "respuesta": translate_text(model_en_es, tokenizer_en_es, response),
    }

    # Traducciones al catalán
    translations_ca = {
        "helsinki": {
            "pregunta": translate_text(model_en_ca, tokenizer_en_ca, question),
            "contexto": translate_text(model_en_ca, tokenizer_en_ca, context),
            "respuesta": translate_text(model_en_ca, tokenizer_en_ca, response),
        },
        "es_ca": {
            "pregunta": translate_text(model_es_ca, tokenizer_es_ca, translations_es["pregunta"]),
            "contexto": translate_text(model_es_ca, tokenizer_es_ca, translations_es["contexto"]),
            "respuesta": translate_text(model_es_ca, tokenizer_es_ca, translations_es["respuesta"]),
        }
    }

    data[i] = {
        "en": {
            "pregunta": question,
            "contexto": context,
            "respuesta": response
        },
        "es": translations_es,
        "ca": translations_ca
    }

# Guardar resultados en un archivo JSON
with open("00_marc/traducciones_completas.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("✅ Traducción completa guardada en '00_marc/traducciones_completas.json'")
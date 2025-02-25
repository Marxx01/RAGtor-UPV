import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Seleccionar el mejor dispositivo disponible
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Cargar dataset
ds = load_dataset("9wimu9/wiki_support_docs_en", split="train")

# Cargar modelo y tokenizer de NLLB-200
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

# Definir códigos de idioma para NLLB-200
lang_code_es = "spa_Latn"  # Español
lang_code_ca = "cat_Latn"  # Catalán

# Diccionario para almacenar los resultados
data = {}

# Procesar datos en un solo bucle
for i, row in enumerate(ds):
    # Obtener datos originales en inglés
    question = row["question"]
    context = " ".join(row["support_documents"]) if isinstance(row["support_documents"], list) else row["support_documents"]
    response = row["answer"]

    # Tokenizar entrada
    inputs = tokenizer([question, context, response], return_tensors="pt", truncation=True, padding=True).to(device)

    # Generar traducciones para español
    inputs['decoder_start_token_id'] = tokenizer.convert_tokens_to_ids([lang_code_es])[0]  # Forzar token de idioma español
    outputs_es = model.generate(**inputs)
    translations_es = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs_es]

    # Generar traducciones para catalán
    inputs['decoder_start_token_id'] = tokenizer.convert_tokens_to_ids([lang_code_ca])[0]  # Forzar token de idioma catalán
    outputs_ca = model.generate(**inputs)
    translations_ca = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs_ca]

    # Guardar en diccionario
    data[f"i_{i}"] = {
        "original_en": {
            "pregunta": question,
            "contexto": context,
            "respuesta": response
        },
        "traduccion_es": {
            "pregunta": translations_es[0],
            "contexto": translations_es[1],
            "respuesta": translations_es[2]
        },
        "traduccion_ca": {
            "pregunta": translations_ca[0],
            "contexto": translations_ca[1],
            "respuesta": translations_ca[2]
        }
    }

    # Procesar solo los primeros 5 elementos como prueba
    if i >= 4:
        break

# Guardar resultados en un archivo JSON
with open("00_marc/traducciones_nllb200.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("Traducción completada y guardada en 'traducciones_nllb200.json'")
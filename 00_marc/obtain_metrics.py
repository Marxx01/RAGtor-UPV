from get_response import PoliGPT
import rag_metrics as rm
import json
import sys

if len(sys.argv) > 1:
    # Si se han pasado argumentos, asignar los valores a las variables
    MODEL_NAME = sys.argv[1]
    CHUNK_SIZE = int(sys.argv[2])
    CHUNK_OVERLAP = int(sys.argv[3])
    K = int(sys.argv[4])
else:
    MODEL_NAME = "sentence-transformers/LaBSE"
    CHUNK_SIZE = 300
    CHUNK_OVERLAP = 100
    K = 4


poligpt = PoliGPT(faiss_index_dir='./01_data/project_faiss', model_name= MODEL_NAME)

def read_json(json_path):
    preguntas, respuestas_modelo = [], []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:
        for preg in entry["preguntas"]:
            preguntas.append(preg["pregunta"])
            respuestas_modelo.append(preg["respuesta"])

    return preguntas, respuestas_modelo

preguntas, respuestas_modelo = read_json("./01_data/preguntas.json")

respuestas = []
for pregunta in preguntas:  
    respuesta = poligpt.query_poligpt(pregunta, k_context=K)
    if 'error' in respuesta:
        print(f"Error en la pregunta: {pregunta}")
        continue
    if respuesta['response'] != 'No dispongo de información suficiente en el contexto proporcionado.':
        respuestas.append(respuesta)


print(f"Número de respuestas obtenidas: {len(respuestas)}")

def extract_context_data(respuestas):
    """
    Toma la salida de PoliGPT (lista de dicts con 'context_used') y
    devuelve:
      - all_contexts: para cada respuesta, la lista de textos de contexto.
      - all_metadata: para cada respuesta, la lista de tuplas (doc_id, página).
    """
    all_contexts = []
    all_metadata = []
    for rsp in respuestas:
        ctxs = []
        metas = []
        for doc, score in rsp['context_used']:
            # doc es un objeto Document con atributos .metadata y .id
            ctxs.append(doc.page_content)
            # extraemos el id y la página:
            doc_id = getattr(doc, 'id', None)
            page   = doc.metadata.get('page')
            metas.append((doc_id, page))
        all_contexts.append(ctxs)
        all_metadata.append(metas)
    return all_contexts, all_metadata

contextos, metadatos = extract_context_data(respuestas)
respuestas_final = [dic['response'] for dic in respuestas]
preguntas_final = [dic['query'] for dic in respuestas]

def evaluate_with_references(preguntas, respuestas, contextos, preguntas_referencia, model_name):
    """Aggregate both reference-based and reference-free metrics.  
    Records must include keys:
      - "respuesta" (prediction)
      - "reference" (ground truth answer)
      - "pregunta", "contextos"
    """
    ems, f1_r, rouge_l, gs, f1s, sims, c_cxt_anw, c_qst_anw, c_qst_cxt, l2_cxt_anw, l2_qst_anw, l2_qst_cxt, c_ref_anw, l2_ref_anw = [], [], [], [], [], [], [], [], [], [], [], [], [], []
    for pred, ref, ctx, q in zip(respuestas, preguntas_referencia, contextos, preguntas):
        ems.append(rm.exact_match_score(pred, ref))
        f1_r.append(rm.token_f1_score(pred, ref))
        rouge_l.append(rm.rouge_l_score(pred, ref))
        gs.append(rm.grounding_score(pred, ctx))
        f1s.append(rm.context_overlap_f1(pred, ctx))
        sims.append(rm.question_context_similarity(q, ctx))
        c_cxt_anw.append(rm.cosine_similarity_score_context(pred, ctx, model_name))
        c_qst_anw.append(rm.cosine_similarity_score(pred, q, model_name))
        c_qst_cxt.append(rm.cosine_similarity_score_context(q, ctx, model_name))
        c_ref_anw.append(rm.cosine_similarity_score(ref, pred, model_name))
        l2_cxt_anw.append(rm.avg_l2_distance_context(pred, ctx, model_name))
        l2_qst_anw.append(rm.avg_l2_distance(pred, q, model_name))
        l2_qst_cxt.append(rm.avg_l2_distance_context(q, ctx, model_name))
        l2_ref_anw.append(rm.avg_l2_distance(ref, pred, model_name))

    n = max(len(ems), 1)
    return {
        "ExactMatch": sum(ems)/n,
        "TokenF1":    sum(f1_r)/n,
        "ROUGE_L":    sum(rouge_l)/n,
        "GroundingScore":  sum(gs)/n,
        "ContextOverlapF1":  sum(f1s)/n,
        "QuestionContextSim": sum(sims)/n,
        "CosineContextAnswer": sum(c_cxt_anw)/n,
        "CosineQuestionAnswer": sum(c_qst_anw)/n,
        "CosineQuestionContext": sum(c_qst_cxt)/n,
        "CosineReferenceAnswer": sum(c_ref_anw)/n,
        "L2ContextAnswer": sum(l2_cxt_anw)/n,
        "L2QuestionAnswer": sum(l2_qst_anw)/n,
        "L2QuestionContext": sum(l2_qst_cxt)/n,
        "L2ReferenceAnswer": sum(l2_ref_anw)/n,
    }

data = evaluate_with_references(preguntas_final, respuestas_final, contextos, respuestas_modelo, model_name=MODEL_NAME)

full_data = {
    "model_name": MODEL_NAME,
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
    "k": K,
    "metrics": data
}

name_safe = MODEL_NAME.split("/")[-1]
full_data["model_name"] = name_safe

with open(f"./00_marc/metrics_eval/metrics_{name_safe}_{CHUNK_SIZE}_{CHUNK_OVERLAP}_{K}.json", "w", encoding="utf-8") as f:
    json.dump(full_data, f, ensure_ascii=False, indent=4)

print("Evaluación completada y guardada en el archivo JSON.")

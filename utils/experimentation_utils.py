import torch
from bert_score import score

import psutil
import os
import time
from pynvml import *
import numpy as p
from typing import Dict

device = "cuda" if torch.cuda.is_available() else "cpu"

def aplanar(l):
        flat = []
        for i in l:
            if isinstance(i, list):
                flat.extend(aplanar(i))
            else:
                flat.append(str(i))
        return flat
    
def calcular_bertscore_listas(predicciones, referencias):
    """
    Compara dos listas de strings usando BERTScore.
    Concatenamos los elementos para obtener una puntuación global del paper.
    """
    if not predicciones or not referencias:
        return 0.0

    predicciones=aplanar(predicciones)

    # Convertimos listas a un único string representativo
    cands = [" ".join(predicciones)]
    refs = [" ".join(referencias)]

    # Calculamos BERTScore (P, R, F1). Usamos F1 como métrica principal.
    # 'lang="en"' es necesario para papers científicos.
    P, R, F1 = score(cands, refs, lang="en", device=device, verbose=False)
    return F1.item()


def calcular_f1_global_card(pred_card: Dict, gt_card: Dict):
    """
    Calcula el F1 global comparando todos los campos de la Model Card.
    """
    scores_por_campo = []
    
    # Definimos los campos que queremos evaluar
    campos_interes = ['methods', 'tasks', 'datasets', 'repo_url', 'category']
    
    for campo in campos_interes:
        val_pred = str(pred_card.get(campo, ""))
        val_gt = str(gt_card.get(campo, ""))
        
        if not val_gt and not val_pred:
            continue # Si ambos están vacíos, no penalizamos
            
        # Reutilizamos tu función de BERTScore para cada campo
        # (Asumiendo que devuelve un valor entre 0 y 1)
        F1 = score(cands, refs, lang="en", device=device, verbose=False)
        scores_por_campo.append(F1)
    
    # El F1 Global es la media de todos los campos evaluados
    return np.mean(scores_por_campo) if scores_por_campo else 0.0


def get_process_info():
    process = psutil.Process(os.getpid())
    # Memoria RAM usada por el script en Megabytes
    mem_usage = process.memory_info().rss / (1024 * 1024)
    # Carga de CPU (porcentaje)
    cpu_usage = psutil.cpu_percent(interval=None)
    return mem_usage, cpu_usage



def get_gpu_usage():
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0) # La primera GPU
        info = nvmlDeviceGetMemoryInfo(handle)
        used_vram = info.used / (1024**2) # Convertir a MB
        nvmlShutdown()
        return used_vram
    except Exception as e:
        print(f"Error leyendo GPU: {e}")
        return 0


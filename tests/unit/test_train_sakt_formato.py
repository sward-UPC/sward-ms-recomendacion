"""Tests puros de la variante format-aware del entrenamiento SAKT.

Solo cubren la lógica de PARSEO e INDEXADO (funciones puras de
``training/train_sakt_formato.py``). No importan torch ni pyKT, así corren en
cualquier entorno de CI sin GPU ni dependencias pesadas.

El objetivo de tesis que validan: que el skill se construya como el par
compuesto (concepto, formato), con degradación a "generico" cuando falta el
formato, de forma determinista.
"""

import importlib.util
import sys
from pathlib import Path

# Importa el módulo de entrenamiento por ruta (training/ no es paquete instalado)
# evitando ejecutar main() y sin requerir torch para estas funciones puras.
_MOD_PATH = Path(__file__).resolve().parents[2] / "training" / "train_sakt_formato.py"
_spec = importlib.util.spec_from_file_location("train_sakt_formato", _MOD_PATH)
tsf = importlib.util.module_from_spec(_spec)
sys.modules["train_sakt_formato"] = tsf
_spec.loader.exec_module(tsf)


# ---------------------------------------------------------------------------
# normalizar_formato
# ---------------------------------------------------------------------------


def test_normalizar_formato_none_es_generico():
    assert tsf.normalizar_formato(None) == tsf.FORMATO_GENERICO


def test_normalizar_formato_vacio_es_generico():
    assert tsf.normalizar_formato("") == tsf.FORMATO_GENERICO
    assert tsf.normalizar_formato("   ") == tsf.FORMATO_GENERICO


def test_normalizar_formato_canoniza_mayusculas_y_espacios():
    assert tsf.normalizar_formato("  Quiz ") == "quiz"
    assert tsf.normalizar_formato("LECTURA") == "lectura"


# ---------------------------------------------------------------------------
# skill_compuesto
# ---------------------------------------------------------------------------


def test_skill_compuesto_distingue_formato():
    leido = tsf.skill_compuesto("AVL", "lectura")
    practicado = tsf.skill_compuesto("AVL", "quiz")
    assert leido != practicado
    assert leido == f"AVL{tsf.SEP_TOKEN}lectura"
    assert practicado == f"AVL{tsf.SEP_TOKEN}quiz"


def test_skill_compuesto_sin_formato_usa_generico():
    assert tsf.skill_compuesto("AVL", None) == f"AVL{tsf.SEP_TOKEN}generico"


def test_skill_compuesto_respeta_capitalizacion_del_concepto():
    # El concepto NO se baja a minúsculas (llega tal cual de Moodle).
    assert tsf.skill_compuesto("Grafos", "Quiz") == f"Grafos{tsf.SEP_TOKEN}quiz"


# ---------------------------------------------------------------------------
# cargar_secuencias + construir_indice (índice compuesto)
# ---------------------------------------------------------------------------


def _data_sintetica():
    # Dos estudiantes; "AVL" aparece en dos formatos → dos skills compuestos.
    return [
        {
            "estudiante_id": "e1",
            "concepto": "AVL",
            "correcta": True,
            "orden": "2026-01-01",
            "tipo_recurso": "lectura",
        },
        {
            "estudiante_id": "e1",
            "concepto": "AVL",
            "correcta": False,
            "orden": "2026-01-02",
            "tipo_recurso": "quiz",
        },
        {
            "estudiante_id": "e1",
            "concepto": "Grafos",
            "correcta": True,
            "orden": "2026-01-03",
            "tipo_recurso": "quiz",
        },
        # e2 trae una interacción SIN tipo_recurso → debe caer en "generico".
        {
            "estudiante_id": "e2",
            "concepto": "AVL",
            "correcta": True,
            "orden": "2026-01-01",
        },
        {
            "estudiante_id": "e2",
            "concepto": "Grafos",
            "correcta": False,
            "orden": "2026-01-02",
            "tipo_recurso": "quiz",
        },
    ]


def test_cargar_secuencias_agrupa_por_estudiante():
    seqs = tsf.cargar_secuencias(_data_sintetica())
    assert len(seqs) == 2  # e1 y e2


def test_cargar_secuencias_ordena_por_orden():
    data = [
        {
            "estudiante_id": "e1",
            "concepto": "AVL",
            "correcta": True,
            "orden": "2026-01-03",
            "tipo_recurso": "quiz",
        },
        {
            "estudiante_id": "e1",
            "concepto": "AVL",
            "correcta": False,
            "orden": "2026-01-01",
            "tipo_recurso": "lectura",
        },
    ]
    seqs = tsf.cargar_secuencias(data)
    # La de orden más temprano (lectura) va primero.
    assert seqs[0][0][0] == f"AVL{tsf.SEP_TOKEN}lectura"
    assert seqs[0][1][0] == f"AVL{tsf.SEP_TOKEN}quiz"


def test_indice_compuesto_separa_concepto_por_formato():
    seqs = tsf.cargar_secuencias(_data_sintetica())
    idx = tsf.construir_indice(seqs)
    # AVL aparece como lectura, quiz y generico → 3 skills solo para AVL.
    assert f"AVL{tsf.SEP_TOKEN}lectura" in idx
    assert f"AVL{tsf.SEP_TOKEN}quiz" in idx
    assert f"AVL{tsf.SEP_TOKEN}generico" in idx
    assert f"Grafos{tsf.SEP_TOKEN}quiz" in idx
    # 4 skills compuestos en total.
    assert len(idx) == 4


def test_indice_es_determinista_y_contiguo():
    seqs = tsf.cargar_secuencias(_data_sintetica())
    idx = tsf.construir_indice(seqs)
    # Índices 0..n-1 sin huecos.
    assert sorted(idx.values()) == list(range(len(idx)))
    # Orden alfabético → reproducible entre corridas.
    claves_ordenadas = sorted(idx, key=lambda k: idx[k])
    assert claves_ordenadas == sorted(idx.keys())


def test_construir_indice_formatos():
    seqs = tsf.cargar_secuencias(_data_sintetica())
    idx = tsf.construir_indice(seqs)
    formatos = tsf.construir_indice_formatos(idx)
    assert set(formatos.keys()) == {"lectura", "quiz", "generico"}
    assert sorted(formatos.values()) == list(range(len(formatos)))


def test_sin_formato_colapsa_a_un_solo_skill_por_concepto():
    # Si NINGUNA interacción trae tipo_recurso, la variante degrada al caso
    # de producción: un skill por concepto (todos "<concepto>||generico").
    data = [
        {"estudiante_id": "e1", "concepto": "AVL", "correcta": True, "orden": "1"},
        {"estudiante_id": "e1", "concepto": "Grafos", "correcta": False, "orden": "2"},
    ]
    seqs = tsf.cargar_secuencias(data)
    idx = tsf.construir_indice(seqs)
    assert set(idx.keys()) == {
        f"AVL{tsf.SEP_TOKEN}generico",
        f"Grafos{tsf.SEP_TOKEN}generico",
    }

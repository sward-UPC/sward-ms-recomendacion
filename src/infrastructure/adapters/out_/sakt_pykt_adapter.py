import logging
import os
import random
import sys
import tempfile
import time
import types as _types
import zlib
from datetime import timezone

from src.domain.entities.fidelidad_explicacion import (
    CRITERIO_EXHAUSTIVIDAD,
    CRITERIO_SUFICIENCIA,
    CRITERIOS_VALIDOS,
    MOTIVO_DESACTIVADA,
    MOTIVO_ERROR,
    MOTIVO_FIEL,
    MOTIVO_NO_SUPERA_AZAR,
    MOTIVO_POCAS_INTERACCIONES,
    Contrafactual,
    FidelidadExplicacion,
    PruebaFidelidad,
)
from src.domain.entities.prediccion_kt import PrediccionKT
from src.domain.entities.secuencia_interaccion import SecuenciaInteraccion
from src.application.ports.out_.modelo_kt_port import ModeloKTPort

logger = logging.getLogger(__name__)

# El modelo usa 0 como relleno. Borrar una interacción = ponerla en PAD, que es
# como el SAKT representa "posición vacía". Idéntico a evaluation/xai_faithfulness.py.
PAD_TOKEN = 0


# Formatos de entrada de los dos pipelines de entrenamiento del proyecto.
FORMATO_IZQUIERDA = "relleno_izquierda"  # training/train_sakt.py (producción)
FORMATO_DERECHA = "relleno_derecha"  # sward-model-training/train.py (investigación)
FORMATOS_VALIDOS = (FORMATO_IZQUIERDA, FORMATO_DERECHA)


def detectar_formato_entrada(checkpoint: dict, forzado: str = "") -> str:
    """Formato con que se entrenó el checkpoint, para servirlo igual.

    Orden de prioridad:
      1. `forzado` (variable SAKT_FORMATO_ENTRADA), para casos excepcionales.
      2. La clave `formato_entrada` que ambos pipelines escriben desde ahora.
      3. Para checkpoints anteriores, la huella de sus claves: train.py guarda
         `val_auc` y `epoch`; train_sakt.py guarda `trained_at` y `test_auc`.
      4. Si no hay señal, el formato histórico del adaptador (izquierda), para
         no alterar el comportamiento de modelos que ya estaban en servicio.
    """
    if forzado:
        if forzado not in FORMATOS_VALIDOS:
            raise ValueError(
                f"SAKT_FORMATO_ENTRADA inválido: {forzado!r}. "
                f"Valores posibles: {', '.join(FORMATOS_VALIDOS)}"
            )
        return forzado
    declarado = checkpoint.get("formato_entrada")
    if declarado in FORMATOS_VALIDOS:
        return declarado
    if "val_auc" in checkpoint and "trained_at" not in checkpoint:
        return FORMATO_DERECHA
    return FORMATO_IZQUIERDA


def _confianza_binaria(prob: float) -> float:
    """Confianza de la predicción binaria = |p - 0.5| * 2 ∈ [0, 1].

    Comprehensiveness se define sobre la probabilidad de la *clase predicha*.
    Centrar en 0.5 evita que la métrica dependa de cuál clase ganó. Misma
    definición que la evaluación offline, para que los números sean comparables.
    """
    return abs(prob - 0.5) * 2.0


def _mock_turtle() -> None:
    """pyKT bug: qdkt.py importa turtle (requiere Tk). Mock antes de que pyKT cargue."""
    if "turtle" not in sys.modules:
        _mock = _types.ModuleType("turtle")
        _mock.forward = lambda *a, **kw: None
        sys.modules["turtle"] = _mock


def leer_info_modelo() -> dict:
    """Metadata REAL del modelo entrenado, leída del artefacto en S3 (no hardcode).

    Fuente de verdad para el panel admin: la fecha de último reentrenamiento sale
    del `LastModified` del objeto en S3 y los hiperparámetros/métricas del propio
    checkpoint (consistentes con el modelo desplegado). No construye la red SAKT;
    solo abre el dict del checkpoint, así que es liviano.
    """
    import boto3
    import torch

    from src.infrastructure.config.settings import settings

    s3 = boto3.client("s3", region_name=settings.aws_region)
    head = s3.head_object(
        Bucket=settings.aws_s3_model_bucket, Key=settings.sakt_model_s3_key
    )
    actualizado_en = head["LastModified"].astimezone(timezone.utc).isoformat()
    tam_bytes = int(head["ContentLength"])

    # Cache local; re-descarga si cambió el tamaño (p.ej. tras un reentrenamiento).
    local_path = os.path.join(tempfile.gettempdir(), "sakt_info.pth")
    if not os.path.exists(local_path) or os.path.getsize(local_path) != tam_bytes:
        s3.download_file(
            settings.aws_s3_model_bucket, settings.sakt_model_s3_key, local_path
        )

    ckpt = torch.load(local_path, map_location="cpu", weights_only=True)
    concept_index = ckpt.get("concept_index") or {}
    return {
        "n_skills": ckpt.get("n_skills"),
        "n_conceptos": len(concept_index) or ckpt.get("n_skills"),
        "seq_len": ckpt.get("seq_len"),
        "emb_size": ckpt.get("emb_size"),
        "n_heads": ckpt.get("n_heads"),
        "n_layers": ckpt.get("n_layers"),
        "dropout": ckpt.get("dropout"),
        "learning_rate": ckpt.get("learning_rate"),
        "epochs": ckpt.get("epochs"),
        "test_auc": ckpt.get("test_auc"),
        "n_estudiantes": ckpt.get("n_estudiantes"),
        "n_muestras": ckpt.get("n_muestras"),
        "entrenado_en": ckpt.get("trained_at") or actualizado_en,
        "actualizado_en": actualizado_en,
        "tam_bytes": tam_bytes,
    }


class SaktPyktAdapter(ModeloKTPort):
    """Adaptador real del SAKT: carga el checkpoint desde S3 con torch/pyKT e infiere.

    Implementa ``ModeloKTPort`` moviendo aquí, sin cambios, toda la lógica de
    inferencia y de captura de atención que vivía en el dominio. Si la carga del
    artefacto falla, cae a una predicción mock (mismo comportamiento que antes).
    """

    def __init__(self, version: str = "v1.0"):
        self.version = version
        self._mock = False
        self._model = None
        self._seq_len = 200
        # Mapa concepto(str)→índice entero del modelo. Vacío para modelos legacy
        # (assist2015, cuyos conceptos ya son ints); poblado para modelos Moodle.
        self._concept_index: dict[str, int] = {}
        self._formato = FORMATO_IZQUIERDA
        self._inverso_conceptos: dict[int, str] | None = None
        self._cargar_modelo()

    def _cargar_modelo(self) -> None:
        try:
            import torch
            import boto3

            from src.infrastructure.config.settings import settings

            s3 = boto3.client("s3", region_name=settings.aws_region)
            local_path = os.path.join(tempfile.gettempdir(), f"sakt_{self.version}.pth")
            if not os.path.exists(local_path):
                logger.info(
                    "Descargando modelo desde S3 | key=%s", settings.sakt_model_s3_key
                )
                s3.download_file(
                    settings.aws_s3_model_bucket, settings.sakt_model_s3_key, local_path
                )

            # El checkpoint contiene solo tensores y primitivas Python → weights_only=True es seguro
            checkpoint = torch.load(local_path, map_location="cpu", weights_only=True)

            n_skills = checkpoint["n_skills"]
            seq_len = checkpoint["seq_len"]
            emb_size = checkpoint["emb_size"]
            n_heads = checkpoint["n_heads"]
            dropout = checkpoint["dropout"]
            n_layers = checkpoint["n_layers"]
            self._seq_len = seq_len
            # Índice de conceptos del modelo (si fue entrenado con conceptos Moodle).
            self._concept_index = checkpoint.get("concept_index", {}) or {}
            self._formato = detectar_formato_entrada(
                checkpoint, settings.sakt_formato_entrada
            )

            _mock_turtle()

            import pykt.models.utils as _pykt_utils
            from pykt.models.sakt import SAKT, Blocks
            from pykt.models.utils import ut_mask

            _pykt_utils.device = "cpu"

            # Monkey-patch del bloque de atención para CAPTURAR los pesos (el forward
            # de pyKT los descarta) y aceptar máscara de relleno, igual que el
            # forward de sward-model-training/train.py. Con key_padding_mask=None
            # es idéntico al stock, así que el formato histórico no cambia.
            def _blocks_forward_capture(
                self, q=None, k=None, v=None, key_padding_mask=None
            ):

                q, k, v = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
                causal_mask = ut_mask(seq_len=k.shape[0])
                attn_emb, attn_w = self.attn(
                    q,
                    k,
                    v,
                    attn_mask=causal_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=True,
                )
                self._last_attn = attn_w.detach()  # (batch, tgt_len, src_len)
                attn_emb = self.attn_dropout(attn_emb)
                attn_emb, q = attn_emb.permute(1, 0, 2), q.permute(1, 0, 2)
                attn_emb = self.attn_layer_norm(q + attn_emb)
                emb = self.FFN(attn_emb)
                emb = self.FFN_dropout(emb)
                emb = self.FFN_layer_norm(attn_emb + emb)
                return emb

            def _sakt_forward(self, q, r, qry, qtest=False, key_padding_mask=None):
                qshftemb, xemb = self.base_emb(q, r, qry)
                for i in range(self.num_en):
                    xemb = self.blocks[i](
                        qshftemb, xemb, xemb, key_padding_mask=key_padding_mask
                    )
                p = torch.sigmoid(self.pred(self.dropout_layer(xemb))).squeeze(-1)
                return p if not qtest else (p, xemb)

            Blocks.forward = _blocks_forward_capture
            SAKT.forward = _sakt_forward

            model = SAKT(
                num_c=n_skills,
                seq_len=seq_len,
                emb_size=emb_size,
                num_attn_heads=n_heads,
                dropout=dropout,
                num_en=n_layers,
                emb_type="qid",
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            self._model = model
            logger.info(
                "Modelo SAKT cargado | version=%s n_skills=%d seq_len=%d emb_size=%d",
                self.version,
                n_skills,
                seq_len,
                emb_size,
            )
            # print() para que quede visible en CloudWatch (el logger custom no
            # siempre se captura; uvicorn sí captura stdout).
            print(
                f"[SAKT] Modelo REAL cargado desde S3 | n_skills={n_skills} "
                f"seq_len={seq_len} formato_entrada={self._formato}",
                flush=True,
            )
        except Exception as e:
            logger.error("Error cargando SAKT, usando mock: %s", e)
            print(f"[SAKT] FALLBACK A MOCK (no se pudo cargar de S3): {e}", flush=True)
            self._mock = True

    def predecir_dominio(self, secuencia: SecuenciaInteraccion) -> PrediccionKT:
        if self._mock or not secuencia.concepto_ids:
            return self._mock_prediccion(secuencia)
        return self._real_prediccion(secuencia)

    def leer_info(self) -> dict:
        return leer_info_modelo()

    def _mock_prediccion(self, secuencia: SecuenciaInteraccion) -> PrediccionKT:
        if not secuencia.respuestas_correctas:
            prob = 0.5
        else:
            prob = sum(secuencia.respuestas_correctas) / len(
                secuencia.respuestas_correctas
            )
        n = len(secuencia.concepto_ids)
        pesos = [1.0 / n if n > 0 else 0.0] * n
        return PrediccionKT(
            estudiante_id=secuencia.estudiante_id,
            curso_id=secuencia.curso_id,
            probabilidad_dominio=round(prob, 4),
            confianza=0.7 if n >= 5 else 0.4,
            pesos_atencion=pesos,
        )

    def _real_prediccion(self, secuencia: SecuenciaInteraccion) -> PrediccionKT:
        try:
            # Mapear conceptos→índices enteros que entiende el modelo:
            #  - con concept_index (modelo Moodle): traduce la sección; omite desconocidos.
            #  - sin índice (legacy assist2015): los conceptos ya son string-ints.
            pares: list[tuple[int, int]] = []
            for concepto, correcta in zip(
                secuencia.concepto_ids, secuencia.respuestas_correctas
            ):
                if self._concept_index:
                    idx = self._concept_index.get(str(concepto))
                    if idx is None:
                        continue
                elif str(concepto).lstrip("-").isdigit():
                    idx = int(concepto)
                else:
                    continue
                pares.append((idx, 1 if correcta else 0))

            if len(pares) < 2:
                return self._mock_prediccion(secuencia)
            concepts = [p[0] for p in pares]
            responses = [p[1] for p in pares]

            seq_len = self._seq_len
            concepts = concepts[-seq_len:]
            responses = responses[-seq_len:]
            L = len(concepts)

            # SAKT input format (igual que train.py):
            #   q   = past concepts [0..L-2]
            #   r   = past responses [0..L-2]
            #   qry = shifted concepts [1..L-1]  ← output[-1] = P(correct para concepts[-1])
            q = concepts[:-1]
            r = responses[:-1]
            qry = concepts[1:]

            prob = float(self._inferir([q], [r], qry)[0])

            # La atención se extrae AQUÍ, antes de cualquier otra pasada: el
            # monkey-patch guarda solo la última, y la verificación la pisaría.
            pesos = self._extraer_atencion(L - 1)

            # Aporte: comprobar si esa atención de verdad gobierna la predicción
            # sobre ESTA secuencia, antes de que el sistema la presente como el
            # motivo de la recomendación.
            fidelidad = self._verificar_fidelidad(q, r, qry, pesos, prob, secuencia)

            return PrediccionKT(
                estudiante_id=secuencia.estudiante_id,
                curso_id=secuencia.curso_id,
                probabilidad_dominio=round(min(max(prob, 0.0), 1.0), 4),
                # Con la verificación activa la confianza es evidencia medida; con
                # la verificación apagada se conserva el valor histórico, lo que
                # permite comparar ambas versiones sin tocar el código.
                confianza=(
                    fidelidad.confianza
                    if fidelidad.motivo != MOTIVO_DESACTIVADA
                    else 0.85
                ),
                pesos_atencion=pesos,
                fidelidad=fidelidad,
            )
        except Exception as e:
            logger.error("Inferencia real falló, usando mock: %s", e)
            return self._mock_prediccion(secuencia)

    # ── Formato de entrada ───────────────────────────────────────────────────
    #
    # El proyecto tiene dos pipelines de entrenamiento que NO preparan la entrada
    # igual: training/train_sakt.py (el del GitHub Action de producción) rellena
    # por la izquierda y sward-model-training/train.py (el de investigación)
    # rellena por la derecha con máscara de relleno. Como SAKT tiene embedding de
    # posición, servir un checkpoint con el formato del otro pipeline pone las
    # interacciones reales en posiciones que el modelo nunca vio y, con relleno
    # por la izquierda sin máscara, hace que la atención caiga sobre el relleno.
    # Medido con un checkpoint de train.py servido por la izquierda: el 85% de la
    # atención iba al relleno y el AUC del último paso bajaba de 0.676 a 0.604.
    # Por eso el formato no se asume: se lee del checkpoint.

    def _tensores(self, qs: list, rs: list, qry: list, borrados: list | None = None):
        """Arma el lote en el formato con que se entrenó el checkpoint.

        `qs` y `rs` son listas de filas, una por variante. `qry` es único y se
        replica: la consulta —lo que se predice— nunca se perturba. `borrados`,
        si se da, trae por fila el conjunto de índices del pasado que se borran.

        Devuelve (q, r, qry, key_padding_mask, posición a leer).
        """
        import torch

        n = len(qry)
        relleno = [PAD_TOKEN] * (self._seq_len - n)

        if self._formato == FORMATO_DERECHA:
            q_t = torch.LongTensor([fila + relleno for fila in qs])
            r_t = torch.LongTensor([fila + relleno for fila in rs])
            qry_t = torch.LongTensor([list(qry) + relleno] * len(qs))
            kpm = torch.zeros(len(qs), self._seq_len, dtype=torch.bool)
            kpm[:, n:] = True
            for i, indices in enumerate(borrados or []):
                for j in indices:
                    # La posición 0 no se enmascara: bajo máscara causal es la
                    # única clave visible para la consulta 0, y dejarla sin
                    # claves produce NaN que se propaga por las capas. Su
                    # contenido ya está en PAD, así que el borrado se aplica igual.
                    if j != 0:
                        kpm[i, j] = True
            return q_t, r_t, qry_t, kpm, n - 1

        # Formato histórico de producción: relleno por la izquierda, sin máscara.
        # Se conserva tal cual para no alterar los modelos que ya lo usan.
        q_t = torch.LongTensor([relleno + fila for fila in qs])
        r_t = torch.LongTensor([relleno + fila for fila in rs])
        qry_t = torch.LongTensor([relleno + list(qry)] * len(qs))
        return q_t, r_t, qry_t, None, -1

    def _inferir(self, qs: list, rs: list, qry: list, borrados: list | None = None):
        """Una sola pasada por lotes; devuelve la probabilidad del último paso real."""
        import torch

        q_t, r_t, qry_t, kpm, pos = self._tensores(qs, rs, qry, borrados)
        with torch.no_grad():
            # pyKT SAKT.forward ya aplica sigmoid internamente.
            out = self._model(q_t, r_t, qry_t, key_padding_mask=kpm)
        return out[:, pos].tolist()

    def _extraer_atencion(self, n: int) -> list:
        """Pesos con que el último paso real atendió a cada interacción pasada.

        Fallback a uniforme si no se capturó nada; la verificación posterior no
        encontrará entonces ventaja sobre el azar, que es lo correcto: una
        atención uniforme no señala nada.
        """
        pesos = [1.0 / n] * n
        try:
            attn = self._model.blocks[-1]._last_attn  # (batch, seq_len, seq_len)
            if self._formato == FORMATO_DERECHA:
                fila = attn[0, n - 1, :n].tolist()
            else:
                fila = attn[0, -1, -n:].tolist()
            total = sum(fila)
            if total > 0:
                pesos = [round(w / total, 4) for w in fila]
        except Exception as e:
            logger.warning("No se pudo extraer atención real: %s", e)
        return pesos

    # ── Verificación de fidelidad (ERASER en línea) ─────────────────────────
    #
    # Traslada al momento de inferir la comprobación que el proyecto solo hacía
    # offline en evaluation/xai_faithfulness.py. Medida con el formato de entrada
    # correcto, la atención cruda de SAKT no supera de forma consistente al
    # borrado aleatorio, así que presentarla siempre como "el motivo" de la
    # recomendación es afirmar de más. Esta verificación decide, por instancia,
    # si hay evidencia para afirmarlo.

    def _rng_estable(self, secuencia: SecuenciaInteraccion, n_pasado: int):
        """Generador determinista por estudiante, curso y largo de secuencia.

        Sin esto, dos cargas seguidas de la misma pantalla podrían mostrar una
        explicación y luego abstenerse por puro azar del muestreo. Un sistema que
        cambia de opinión al recargar no es creíble. La semilla cambia sola
        cuando el estudiante suma una interacción, que es cuando la explicación
        SÍ debe poder cambiar.
        """
        semilla = zlib.crc32(
            f"{secuencia.estudiante_id}|{secuencia.curso_id}|{n_pasado}".encode()
        )
        return random.Random(semilla)

    def _verificar_fidelidad(
        self,
        q: list,
        r: list,
        qry: list,
        pesos: list,
        prob_base: float,
        secuencia: SecuenciaInteraccion,
    ) -> FidelidadExplicacion:
        """Contrasta la atención contra el azar con las dos pruebas de ERASER.

        - Suficiencia: se conserva SOLO el top-k de atención; gana si la
          predicción pierde menos confianza que conservando k al azar.
        - Exhaustividad: se BORRA el top-k; gana si la predicción pierde más
          confianza que borrando k al azar.

        Cada prueba reporta la fracción de sorteos que la atención gana. El
        criterio configurado decide si se da un motivo; el contrafactual solo se
        adjunta si pasa la exhaustividad, porque afirma necesidad.

        Todas las variantes —dos pruebas, sus sorteos y el contrafactual— van en
        UNA pasada por lotes, lo que mantiene el costo dentro de los 500 ms del
        RF-004-05.
        """
        from src.infrastructure.config.settings import settings

        n_pasado = len(q)
        criterio = settings.xai_criterio_fidelidad
        if criterio not in CRITERIOS_VALIDOS:
            raise ValueError(
                f"XAI_CRITERIO_FIDELIDAD inválido: {criterio!r}. "
                f"Valores posibles: {', '.join(CRITERIOS_VALIDOS)}"
            )

        if not settings.xai_verificacion_activa:
            return FidelidadExplicacion.no_verificada(
                MOTIVO_DESACTIVADA, n_pasado, criterio
            )

        k_suf = settings.xai_k_suficiencia
        k_exh = settings.xai_k_exhaustividad
        n_aleatorios = settings.xai_n_aleatorios

        # Tiene que quedar al menos una interacción fuera del top-k en cada prueba:
        # si k cubriera todo el pasado, perturbar «lo atendido» y perturbar «todo»
        # serían lo mismo y no habría contra qué comparar.
        if n_pasado < max(settings.xai_min_interacciones, k_suf + 1, k_exh + 1):
            return FidelidadExplicacion.no_verificada(
                MOTIVO_POCAS_INTERACCIONES, n_pasado, criterio
            )

        inicio = time.perf_counter()
        try:
            orden = sorted(range(n_pasado), key=lambda i: pesos[i], reverse=True)
            rng = self._rng_estable(secuencia, n_pasado)
            todas = set(range(n_pasado))

            # Conjuntos a BORRAR, fila por fila, en este orden:
            #   exhaustividad: top-k, luego n sorteos
            #   suficiencia:   todo menos top-k, luego todo menos k al azar
            borrar_exh = [set(orden[:k_exh])] + [
                set(rng.sample(range(n_pasado), k_exh)) for _ in range(n_aleatorios)
            ]
            conservar_suf = [set(orden[:k_suf])] + [
                set(rng.sample(range(n_pasado), k_suf)) for _ in range(n_aleatorios)
            ]
            borrar_suf = [todas - c for c in conservar_suf]

            filas_q, filas_r, borrados = [], [], []
            for indices in borrar_exh + borrar_suf:
                fq, fr = list(q), list(r)
                for i in indices:
                    fq[i] = PAD_TOKEN
                    fr[i] = PAD_TOKEN
                filas_q.append(fq)
                filas_r.append(fr)
                borrados.append(indices)

            # Última fila: el contrafactual. No borra: invierte la respuesta de la
            # interacción más atendida («¿y si esto hubiera salido al revés?»).
            idx_top = orden[0]
            r_cf = list(r)
            r_cf[idx_top] = 1 - r_cf[idx_top]
            filas_q.append(list(q))
            filas_r.append(r_cf)
            borrados.append(set())

            probs = self._inferir(filas_q, filas_r, qry, borrados)

            conf_base = _confianza_binaria(prob_base)
            perdidas = [conf_base - _confianza_binaria(p) for p in probs]
            bloque = 1 + n_aleatorios
            p_exh, p_suf = perdidas[:bloque], perdidas[bloque : 2 * bloque]

            exhaustividad = PruebaFidelidad(
                criterio=CRITERIO_EXHAUSTIVIDAD,
                k=k_exh,
                perdida_atencion=round(p_exh[0], 4),
                perdida_azar_media=round(sum(p_exh[1:]) / n_aleatorios, 4),
                n_aleatorios=n_aleatorios,
                # Borrar lo atendido debe doler MÁS que borrar al azar.
                supera_azar_en=sum(1 for x in p_exh[1:] if p_exh[0] > x),
            )
            suficiencia = PruebaFidelidad(
                criterio=CRITERIO_SUFICIENCIA,
                k=k_suf,
                perdida_atencion=round(p_suf[0], 4),
                perdida_azar_media=round(sum(p_suf[1:]) / n_aleatorios, 4),
                n_aleatorios=n_aleatorios,
                # Conservar solo lo atendido debe doler MENOS que conservar al azar.
                supera_azar_en=sum(1 for x in p_suf[1:] if p_suf[0] < x),
            )

            umbral = settings.xai_umbral_confianza
            es_suficiente = suficiencia.confianza >= umbral
            es_necesaria = exhaustividad.confianza >= umbral
            es_fiel = (
                es_suficiente if criterio == CRITERIO_SUFICIENCIA else es_necesaria
            )

            nombre = self._nombre_concepto
            top_suf = orden[:k_suf]

            return FidelidadExplicacion(
                criterio=criterio,
                n_pasado=n_pasado,
                umbral=umbral,
                suficiencia=suficiencia,
                exhaustividad=exhaustividad,
                indices_suficientes=list(top_suf) if es_suficiente else [],
                conceptos_suficientes=(
                    [nombre(q[i]) for i in top_suf] if es_suficiente else []
                ),
                motivo=MOTIVO_FIEL if es_fiel else MOTIVO_NO_SUPERA_AZAR,
                latencia_ms=round((time.perf_counter() - inicio) * 1000, 2),
                contrafactual=(
                    Contrafactual(
                        indice=idx_top,
                        concepto=nombre(q[idx_top]),
                        acierto_original=bool(r[idx_top]),
                        probabilidad_original=min(max(prob_base, 0.0), 1.0),
                        probabilidad_contrafactual=min(max(float(probs[-1]), 0.0), 1.0),
                    )
                    if es_necesaria
                    else None
                ),
            )
        except Exception as e:
            # Que falle la verificación no debe tumbar la recomendación: se
            # devuelve sin motivo, que es el lado seguro.
            logger.warning("Verificación de fidelidad falló: %s", e)
            return FidelidadExplicacion.no_verificada(MOTIVO_ERROR, n_pasado, criterio)

    def _nombre_concepto(self, indice: int) -> str:
        """Nombre legible del concepto a partir del índice del modelo.

        Los modelos Moodle traen concept_index (nombre → índice). Los legacy de
        ASSISTments usan enteros como concepto, así que el índice ya es el nombre.
        """
        if self._inverso_conceptos is None:
            self._inverso_conceptos = {v: k for k, v in self._concept_index.items()}
        return str(self._inverso_conceptos.get(indice, indice))

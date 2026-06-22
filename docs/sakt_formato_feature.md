# Feature de modelado: el FORMATO del recurso en el SAKT

> Contribución de modelado de la tesis SWARD. Estado: **experimental**, aislado
> del pipeline de producción (no toca `train_sakt.py` ni `modelo_sakt.py`).

## 1. Motivación

El SAKT desplegado (knowledge tracing sobre pyKT) modela cada interacción del
estudiante como el par **(concepto, acierto)**. Con esa representación el modelo
es ciego al *cómo* se practicó un concepto: una lectura de "AVL" y un quiz de
"AVL" colapsan al mismo *skill*. Pero pedagógicamente no son equivalentes:
distintos estudiantes consolidan mejor en distintos formatos (leer, practicar,
ver video). Si el modelo no ve el formato, no puede capturar esa preferencia ni
explotar la traza atencional para recomendar el formato ideal por alumno.

**Objetivo:** que el SAKT distinga, p.ej., "AVL practicado" de "AVL leído", para
que el motor de recomendación pueda sugerir el formato en el que cada alumno
aprende mejor.

## 2. Enfoque elegido: skill compuesto (concepto × formato)

Se redefine la unidad de conocimiento (*skill*) como el **par compuesto
`(concepto, tipo_recurso)`**. Cada combinación concepto×formato ocupa su propia
fila de embedding del SAKT y participa con identidad propia en el mecanismo de
atención.

Implementación: `training/train_sakt_formato.py`.

- Las claves del índice se serializan como `"<concepto>||<formato>"` (separador
  `SEP_TOKEN = "||"`).
- El formato se normaliza (`strip().lower()`); si la interacción no trae
  `tipo_recurso`, se usa `"generico"`.
- El índice (`concept_index`) se construye ordenado alfabéticamente →
  determinista y reproducible.
- `num_c` del SAKT pasa a ser el nº de skills **compuestos** (no de conceptos).
  Esa es la **única** diferencia estructural relevante con producción: el
  alfabeto de skills crece (a lo sumo `|conceptos| × |formatos|`).

### ¿Por qué es razonable?

1. **Mínimo cambio arquitectónico.** No se altera la red SAKT de pyKT ni el
   formato de secuencia (`q`, `r`, `qry`, padding/shift son idénticos a
   `train_sakt.py`). Solo cambia el vocabulario de skills. Esto reduce el riesgo
   y hace la comparación de AUC contra el modelo base directa y justa.
2. **Expresividad suficiente.** El SAKT ya aprende relaciones entre skills vía
   atención; al separar por formato, puede aprender que el dominio de
   "AVL||quiz" se predice mejor a partir de "AVL||lectura" para ciertos
   estudiantes, capturando la interacción concepto-formato sin features
   adicionales.
3. **Compatibilidad de checkpoint.** Mantiene las mismas claves que el loader
   actual (`n_skills`, `seq_len`, `emb_size`, ..., `concept_index`,
   `model_state_dict`), más banderas nuevas (`formato_aware`, `sep_token`,
   `formato_index`). Un loader que ignore esas banderas seguiría cargando la red;
   solo fallaría el mapeo de inferencia (ver §5).

### Trade-offs del enfoque compuesto

- **Esparcimiento (sparsity).** El espacio de skills se multiplica. Con pocos
  datos por combinación concepto×formato, algunos embeddings quedan poco
  entrenados. Es el principal riesgo y debe vigilarse con el AUC de validación.
- **No comparte conocimiento entre formatos del mismo concepto.** "AVL||quiz" y
  "AVL||lectura" son índices independientes; el modelo no tiene un sesgo
  explícito de que comparten concepto (lo infiere solo por co-ocurrencia en las
  secuencias). Lo aborda mejor la alternativa de §3.

## 3. Alternativa considerada: embedding de formato separado

En lugar de fusionar, se podría mantener `concept_index` solo por concepto y
añadir un **embedding de formato independiente** que se sume/concatene al
embedding de concepto y de respuesta dentro del SAKT:

```
emb_interaccion = emb_concepto + emb_respuesta + emb_formato
```

**Ventajas:** comparte conocimiento entre formatos del mismo concepto (el
embedding de concepto es uno solo), menos sparsity, y permite analizar el
embedding de formato de forma aislada (¿"quiz" se parece a "ejercicio"?).

**Desventajas:**
- Requiere **modificar la arquitectura de pyKT** (subclase de `SAKT` o
  reescribir `forward` para inyectar el tercer embedding), lo que rompe la
  compatibilidad directa de checkpoint y complica el monkey-patch de captura de
  atención que usa `modelo_sakt.py`.
- Mayor superficie de error y de divergencia frente al modelo base.

**Decisión:** para una contribución de tesis defendible y de bajo riesgo se elige
el **skill compuesto** (§2). La alternativa queda documentada como trabajo
futuro: si la sparsity degrada el AUC, migrar al embedding de formato separado.

## 4. Cambio necesario en el pipeline de datos (dependencia abierta)

Hoy el export interno de ms-trazabilidad (`GET /internal/training-data`) entrega
por interacción:

```json
{"estudiante_id": "...", "concepto": "...", "correcta": true, "orden": "..."}
```

**No incluye el formato.** Para entrenar con formato REAL, el export debe añadir
`tipo_recurso` (el dato ya existe en el dominio de ms-trazabilidad, solo no se
serializa en este endpoint):

```json
{"estudiante_id": "...", "concepto": "...", "correcta": true, "orden": "...",
 "tipo_recurso": "quiz"}
```

Mientras ese cambio no exista, `train_sakt_formato.py` corre igual: toda
interacción sin `tipo_recurso` cae en `"generico"`, y la variante **degrada con
seguridad** al comportamiento de un único skill por concepto (equivalente a
producción). Es decir, el feature está listo del lado del modelo y queda
**bloqueado por el export de datos**.

## 5. Carga e inferencia (qué cambiaría en `modelo_sakt.py`)

> **No se modifica `modelo_sakt.py` de producción en esta entrega.** Aquí se
> describe el cambio que requeriría para servir un checkpoint format-aware.

El loader actual mapea concepto→índice así (resumen de `_real_prediccion`):

```python
idx = self._concept_index.get(str(concepto))   # modelo Moodle
```

Para la variante format-aware harían falta dos cambios:

1. **Leer las banderas nuevas del checkpoint** en `_cargar_modelo`:
   ```python
   self._formato_aware = checkpoint.get("formato_aware", False)
   self._sep_token = checkpoint.get("sep_token", "||")
   ```
2. **Construir la clave compuesta en inferencia.** La `SecuenciaInteraccion`
   debería transportar el formato de cada interacción (nuevo campo, p.ej.
   `tipos_recurso: list[str]` paralelo a `concepto_ids`). Entonces:
   ```python
   if self._formato_aware:
       fmt = (tipo or "generico").strip().lower()
       clave = f"{concepto}{self._sep_token}{fmt}"
       idx = self._concept_index.get(clave)
   else:
       idx = self._concept_index.get(str(concepto))
   ```
   Para **predecir el dominio en un formato concreto** (lo que necesita el motor
   de recomendación), el `qry` del último paso se arma con la clave compuesta del
   formato objetivo: así se obtiene `P(acierto | concepto, formato=X)` y se puede
   comparar el dominio esperado entre formatos para un mismo concepto.

La captura de pesos de atención (`Blocks.forward` monkey-patch) **no cambia**:
opera sobre índices de skill, sea el alfabeto compuesto o no.

## 6. Cómo entrenar la variante

```bash
# Requiere pyKT instalado (no disponible localmente; corre en CI/entorno con torch).
DATA_FILE=training_data.json OUT_FILE=model_formato.pth \
  python training/train_sakt_formato.py
```

Hiperparámetros y métrica (AUC-ROC vía Mann-Whitney U sobre el holdout de
validación) son **idénticos** a `train_sakt.py`, de modo que el `test_auc` del
checkpoint format-aware es directamente comparable con el del modelo base. Esa
comparación (AUC base vs. AUC format-aware) es el experimento central de la
contribución.

## 7. Estado de validación

- `python -m py_compile training/train_sakt_formato.py` — **OK**.
- Tests puros de indexado/parseo: `tests/unit/test_train_sakt_formato.py`
  (no requieren torch/pyKT). Cubren normalización de formato, construcción del
  skill compuesto, agrupado/ordenado de secuencias, índice compuesto
  determinista y degradación a `"generico"`.
- **No** se ejecutó entrenamiento (pyKT no está disponible localmente); la lógica
  de torch/pyKT (`main`, `to_tensors` sobre tensores) **no** está validada en
  ejecución, solo por inspección y por paridad con `train_sakt.py`.

"""Verificación de fidelidad de la explicación por atención, en tiempo de inferencia.

Motivación
----------
SAKT expone pesos de atención y el sistema los presentaba como la razón de la
recomendación ("influencia 34%"). Esa es una afirmación causal: dice que esas
interacciones pasadas son las que mueven la predicción. Medirla es justamente lo
que propone ERASER (DeYoung et al., 2020) y lo que advierte la discusión de
"Attention is not Explanation" (Jain & Wallace, 2019): la atención es un peso
interno, no una garantía de que borrar esas interacciones cambie la salida.

La evaluación offline del proyecto encontró que sobre secuencias cortas la
atención quedaba POR DEBAJO del borrado aleatorio, y que incluso en la mejor
configuración solo lo supera en ~70% de las secuencias con k=3. Es decir: en
torno a una de cada tres explicaciones mostradas, el motivo declarado no era
mejor que señalar interacciones al azar.

Estas entidades trasladan esa comprobación del script batch al momento de
explicar, por instancia, para que el sistema solo afirme una causa cuando puede
sostenerla y se abstenga cuando no.
"""

from dataclasses import dataclass, field

# Razones de abstención. Se guardan en la entidad para que el motor de
# explicabilidad pueda redactar un mensaje honesto y distinto en cada caso, en
# lugar de un genérico "no se puede explicar".
MOTIVO_FIEL = "fiel"
MOTIVO_POCAS_INTERACCIONES = "pocas_interacciones"
MOTIVO_NO_SUPERA_AZAR = "no_supera_azar"
MOTIVO_DESACTIVADA = "verificacion_desactivada"
MOTIVO_ERROR = "error_verificacion"


@dataclass
class Contrafactual:
    """Qué habría pasado si la interacción más atendida hubiera salido al revés.

    Se obtiene invirtiendo la respuesta (acierto ↔ error) de esa interacción y
    volviendo a inferir. A diferencia del mapa de calor, esto sí es accionable
    para el estudiante: nombra una interacción concreta y cuantifica su efecto.

    Es además una comprobación de fidelidad por derecho propio: si invertir la
    interacción más atendida no mueve la predicción, esa atención no estaba
    gobernando la decisión.
    """

    indice: int = 0
    concepto: str = ""
    acierto_original: bool = False
    probabilidad_original: float = 0.5
    probabilidad_contrafactual: float = 0.5

    def __post_init__(self) -> None:
        for nombre, valor in (
            ("probabilidad_original", self.probabilidad_original),
            ("probabilidad_contrafactual", self.probabilidad_contrafactual),
        ):
            if not 0.0 <= valor <= 1.0:
                raise ValueError(
                    f"{nombre} debe estar en [0, 1]; recibido: {valor}"
                )
        if self.indice < 0:
            raise ValueError(
                f"El índice del contrafactual no puede ser negativo; "
                f"recibido: {self.indice}"
            )

    @property
    def delta(self) -> float:
        """Cuánto se movería la probabilidad de dominio. Positivo = subiría."""
        return round(self.probabilidad_contrafactual - self.probabilidad_original, 4)

    @property
    def es_relevante(self) -> bool:
        """Si el cambio es lo bastante grande como para valer la pena mostrarlo.

        Por debajo de un punto porcentual el contrafactual es ruido numérico y
        enunciarlo ("subiría de 0.42 a 0.424") resta credibilidad en vez de
        sumarla.
        """
        return abs(self.delta) >= 0.01


@dataclass
class FidelidadExplicacion:
    """Resultado de contrastar la atención contra el borrado aleatorio.

    El procedimiento, por instancia, es el mismo de la evaluación offline:

    1. Se mide la confianza de la predicción con la secuencia completa.
    2. Se borran las `k` interacciones más atendidas y se vuelve a predecir. La
       caída de confianza es la *comprehensiveness* de la atención.
    3. Se repite borrando `k` interacciones al azar, `n_aleatorios` veces.
    4. `confianza` es la fracción de esos borrados aleatorios a los que la
       atención le gana.

    Esa fracción es el complemento de un p-valor de permutación de una cola: 0.5
    significa "la atención no explica mejor que el azar" y 1.0 significa que le
    gana a todos los sorteos. Se usa esa fracción, y no una función arbitraria
    de la diferencia, porque no introduce ninguna constante de escala que haya
    que justificar ante un jurado.
    """

    k: int = 3
    n_pasado: int = 0
    comprehensiveness_atencion: float = 0.0
    comprehensiveness_azar_media: float = 0.0
    n_aleatorios: int = 0
    supera_azar_en: int = 0
    confianza: float = 0.5
    umbral: float = 0.8
    es_fiel: bool = False
    motivo: str = MOTIVO_NO_SUPERA_AZAR
    latencia_ms: float = 0.0
    contrafactual: Contrafactual | None = field(default=None)

    def __post_init__(self) -> None:
        # Invariante de dominio: la confianza es una fracción.
        if not 0.0 <= self.confianza <= 1.0:
            raise ValueError(
                f"La confianza de fidelidad debe estar en [0, 1]; "
                f"recibido: {self.confianza}"
            )
        if not 0.0 <= self.umbral <= 1.0:
            raise ValueError(
                f"El umbral debe estar en [0, 1]; recibido: {self.umbral}"
            )
        if self.supera_azar_en > self.n_aleatorios:
            raise ValueError(
                f"No se puede superar {self.supera_azar_en} sorteos de "
                f"{self.n_aleatorios}."
            )
        # Una explicación fiel exige haberlo verificado de verdad: sin sorteos no
        # hay evidencia, aunque el motivo diga otra cosa.
        if self.es_fiel and self.n_aleatorios == 0:
            raise ValueError(
                "Una fidelidad marcada como fiel requiere al menos un sorteo "
                "aleatorio de contraste."
            )

    @property
    def delta(self) -> float:
        """Ventaja de la atención sobre el azar, en puntos de comprehensiveness."""
        return round(
            self.comprehensiveness_atencion - self.comprehensiveness_azar_media, 4
        )

    @classmethod
    def no_verificada(cls, motivo: str, n_pasado: int = 0) -> "FidelidadExplicacion":
        """Caso en que no se pudo comprobar nada: se asume NO fiel.

        El sentido por defecto importa. Ante la duda el sistema calla, porque el
        daño de afirmar un motivo falso es mayor que el de no dar motivo.
        """
        return cls(
            n_pasado=n_pasado,
            confianza=0.5,
            es_fiel=False,
            motivo=motivo,
        )

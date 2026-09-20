"""Verificación de fidelidad de la explicación por atención, en tiempo de inferencia.

Motivación
----------
SAKT expone pesos de atención y el sistema los presentaba como la razón de la
recomendación. Medido con ERASER (DeYoung et al., 2020) y con el formato de
entrada correcto, las interacciones más atendidas resultan **suficientes pero no
necesarias**:

- *Suficiencia*: conservando solo lo más atendido, la predicción se mantiene
  mejor que conservando interacciones al azar (k=1: p < 0.0001, r = +0.44).
- *Exhaustividad*: quitando lo más atendido, la predicción no cambia más que
  quitando interacciones al azar (k=3: p = 0.65). El modelo tiene evidencia
  redundante en el resto de la secuencia.

La atención señala un buen resumen, no la causa. Por eso el sistema solo puede
afirmar lo que verifica, y lo verifica por instancia:

- si la prueba de suficiencia pasa, puede decir «estas actividades bastan para
  llegar a esta conclusión»;
- solo si además pasa la de exhaustividad puede ofrecer el contrafactual («si
  hubieras acertado…»), que es una afirmación de necesidad;
- si no pasa, se abstiene de dar un motivo.
"""

from dataclasses import dataclass, field

CRITERIO_SUFICIENCIA = "suficiencia"
CRITERIO_EXHAUSTIVIDAD = "exhaustividad"
CRITERIOS_VALIDOS = (CRITERIO_SUFICIENCIA, CRITERIO_EXHAUSTIVIDAD)

# Razones del resultado. Se guardan para que quien presente la explicación pueda
# redactar un mensaje honesto y distinto en cada caso.
MOTIVO_FIEL = "fiel"
MOTIVO_POCAS_INTERACCIONES = "pocas_interacciones"
MOTIVO_NO_SUPERA_AZAR = "no_supera_azar"
MOTIVO_DESACTIVADA = "verificacion_desactivada"
MOTIVO_ERROR = "error_verificacion"


@dataclass
class Contrafactual:
    """Qué habría pasado si la interacción más atendida hubiera salido al revés.

    Se obtiene invirtiendo la respuesta (acierto ↔ error) de esa interacción y
    volviendo a inferir. Es una afirmación de necesidad —«esto cambió el
    resultado»—, así que solo puede acompañar a una explicación cuya prueba de
    exhaustividad pasó.
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
                raise ValueError(f"{nombre} debe estar en [0, 1]; recibido: {valor}")
        if self.indice < 0:
            raise ValueError(
                f"El índice del contrafactual no puede ser negativo; recibido: {self.indice}"
            )

    @property
    def delta(self) -> float:
        """Cuánto se movería la probabilidad de dominio. Positivo = subiría."""
        return round(self.probabilidad_contrafactual - self.probabilidad_original, 4)

    @property
    def es_relevante(self) -> bool:
        """Si el cambio merece mostrarse.

        Por debajo de un punto porcentual es ruido numérico, y enunciarlo
        («subiría de 0.42 a 0.424») resta credibilidad en vez de sumarla.
        """
        return abs(self.delta) >= 0.01


@dataclass
class PruebaFidelidad:
    """Una prueba ERASER sobre una sola secuencia, contrastada contra el azar.

    `perdida_atencion` es la confianza que pierde la predicción al perturbar el
    top-k de atención; `perdida_azar_media`, la que pierde en promedio al
    perturbar k interacciones al azar. Qué perturbación se aplica y qué
    dirección cuenta como victoria dependen del criterio:

    - suficiencia: se CONSERVA solo el top-k; la atención gana si pierde MENOS.
    - exhaustividad: se BORRA el top-k; la atención gana si pierde MÁS.

    `supera_azar_en / n_aleatorios` es el complemento de un p-valor de
    permutación de una cola: 0.5 significa «no explica mejor que el azar». Se usa
    esa fracción como confianza porque no introduce ninguna constante de escala.
    """

    criterio: str = CRITERIO_SUFICIENCIA
    k: int = 1
    perdida_atencion: float = 0.0
    perdida_azar_media: float = 0.0
    n_aleatorios: int = 0
    supera_azar_en: int = 0

    def __post_init__(self) -> None:
        if self.criterio not in CRITERIOS_VALIDOS:
            raise ValueError(f"Criterio de fidelidad inválido: {self.criterio!r}")
        if self.k < 1:
            raise ValueError(f"k debe ser al menos 1; recibido: {self.k}")
        if not 0 <= self.supera_azar_en <= self.n_aleatorios:
            raise ValueError(
                f"No se pueden superar {self.supera_azar_en} sorteos de "
                f"{self.n_aleatorios}."
            )

    @property
    def confianza(self) -> float:
        if self.n_aleatorios == 0:
            return 0.5
        return self.supera_azar_en / self.n_aleatorios

    @property
    def ventaja(self) -> float:
        """Ventaja de la atención sobre el azar. Positiva = la atención explica mejor."""
        diferencia = self.perdida_atencion - self.perdida_azar_media
        if self.criterio == CRITERIO_SUFICIENCIA:
            diferencia = -diferencia
        return round(diferencia, 4)


@dataclass
class FidelidadExplicacion:
    """Veredicto de fidelidad de la explicación para una predicción concreta.

    `criterio` fija qué prueba decide si el sistema puede dar un motivo. Las
    propiedades `es_fiel`, `es_suficiente` y `es_necesaria` se derivan de las
    pruebas en vez de guardarse, para que no puedan contradecirlas.
    """

    criterio: str = CRITERIO_SUFICIENCIA
    n_pasado: int = 0
    umbral: float = 0.8
    suficiencia: PruebaFidelidad | None = None
    exhaustividad: PruebaFidelidad | None = None
    # Posiciones del pasado y nombres de concepto de las interacciones que
    # bastan para la predicción. Solo se informan si la suficiencia se verificó.
    indices_suficientes: list[int] = field(default_factory=list)
    conceptos_suficientes: list[str] = field(default_factory=list)
    motivo: str = MOTIVO_NO_SUPERA_AZAR
    latencia_ms: float = 0.0
    contrafactual: Contrafactual | None = None

    def __post_init__(self) -> None:
        if self.criterio not in CRITERIOS_VALIDOS:
            raise ValueError(f"Criterio de fidelidad inválido: {self.criterio!r}")
        if not 0.0 <= self.umbral <= 1.0:
            raise ValueError(f"El umbral debe estar en [0, 1]; recibido: {self.umbral}")
        for nombre, prueba, esperado in (
            ("suficiencia", self.suficiencia, CRITERIO_SUFICIENCIA),
            ("exhaustividad", self.exhaustividad, CRITERIO_EXHAUSTIVIDAD),
        ):
            if prueba is not None and prueba.criterio != esperado:
                raise ValueError(f"La prueba de {nombre} trae criterio {prueba.criterio!r}.")
        # El sistema no puede nombrar interacciones «suficientes» sin haberlo
        # verificado, ni ofrecer un contrafactual sin haber verificado necesidad.
        if (self.indices_suficientes or self.conceptos_suficientes) and not self.es_suficiente:
            raise ValueError("Se informan interacciones suficientes sin verificar suficiencia.")
        if self.contrafactual is not None and not self.es_necesaria:
            raise ValueError("Un contrafactual requiere haber verificado exhaustividad.")
        if self.motivo in (MOTIVO_FIEL, MOTIVO_NO_SUPERA_AZAR) and (
            (self.motivo == MOTIVO_FIEL) != self.es_fiel
        ):
            raise ValueError(
                f"El motivo {self.motivo!r} contradice el resultado de las pruebas."
            )

    def _pasa(self, prueba: PruebaFidelidad | None) -> bool:
        return prueba is not None and prueba.n_aleatorios > 0 and (
            prueba.confianza >= self.umbral
        )

    @property
    def es_suficiente(self) -> bool:
        return self._pasa(self.suficiencia)

    @property
    def es_necesaria(self) -> bool:
        return self._pasa(self.exhaustividad)

    @property
    def prueba_decisiva(self) -> PruebaFidelidad | None:
        if self.criterio == CRITERIO_SUFICIENCIA:
            return self.suficiencia
        return self.exhaustividad

    @property
    def es_fiel(self) -> bool:
        """Si el sistema puede dar un motivo según el criterio configurado."""
        return self._pasa(self.prueba_decisiva)

    @property
    def confianza(self) -> float:
        prueba = self.prueba_decisiva
        return prueba.confianza if prueba is not None else 0.5

    @classmethod
    def no_verificada(
        cls, motivo: str, n_pasado: int = 0, criterio: str = CRITERIO_SUFICIENCIA
    ) -> "FidelidadExplicacion":
        """Caso en que no se pudo comprobar nada: no se afirma ningún motivo.

        El sentido por defecto importa. Ante la duda el sistema calla, porque el
        daño de afirmar un motivo falso es mayor que el de no dar motivo.
        """
        return cls(criterio=criterio, n_pasado=n_pasado, motivo=motivo)

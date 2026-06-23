from src.application.ports.out_.modelo_kt_port import ModeloKTPort


class ModeloInfoNoDisponibleError(RuntimeError):
    """El artefacto del modelo no se pudo leer desde S3 (mapea a 503)."""


class ConsultarModeloInfoUseCase:
    """Lee la metadata real del modelo SAKT (consumida s2s por el panel admin).

    La metadata describe el artefacto entrenado en S3, así que refleja siempre el
    modelo desplegado. ``es_mock`` indica si ESTE entorno corre el modelo simulado.
    """

    def __init__(self, modelo: ModeloKTPort, *, es_mock: bool):
        self._modelo = modelo
        self._es_mock = es_mock

    def execute(self) -> dict:
        try:
            info = self._modelo.leer_info()
        except Exception as exc:  # noqa: BLE001
            raise ModeloInfoNoDisponibleError(
                f"No se pudo leer el modelo desde S3: {exc}"
            ) from exc
        return {"mock": self._es_mock, **info}

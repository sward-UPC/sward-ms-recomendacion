import logging

import httpx

from src.domain.ports.out_.youtube_client_port import YoutubeClientPort
from src.infrastructure.config.settings import settings

logger = logging.getLogger(__name__)

YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"


class YoutubeRestAdapter(YoutubeClientPort):
    """Busca un video real en YouTube vía la Data API v3.

    Best-effort: sin ``youtube_api_key`` o ante cualquier fallo (red, cuota,
    respuesta vacía) retorna None para no romper la generación de material.
    """

    async def buscar_video(self, query: str) -> dict | None:
        if not settings.youtube_api_key or not query:
            return None
        params = {
            "part": "snippet",
            "type": "video",
            "maxResults": 1,
            "q": query,
            "key": settings.youtube_api_key,
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(YOUTUBE_SEARCH_URL, params=params)
            if r.status_code != 200:
                logger.warning("YouTube respondió %s: %s", r.status_code, r.text[:200])
                return None
            items = r.json().get("items") or []
            if not items:
                return None
            item = items[0]
            video_id = item.get("id", {}).get("videoId")
            if not video_id:
                return None
            titulo = item.get("snippet", {}).get("title", "")
            return {
                "video_id": video_id,
                "titulo": titulo,
                "url": f"https://www.youtube.com/watch?v={video_id}",
            }
        except Exception as e:  # best-effort: cualquier fallo => None
            logger.warning("YouTube falló: %s", e)
            return None

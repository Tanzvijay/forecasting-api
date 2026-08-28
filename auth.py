import os
import secrets

from dotenv import load_dotenv
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

load_dotenv()

API_KEY_HEADER = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="Pass your API key in the **X-API-Key** header.",
)


def _load_api_key() -> str:
    key = os.getenv("Key")              # ← matches your .env
    if not key or not key.strip():
        raise RuntimeError(
            "Key is not set. "
            "Add Key=<your-key> to your .env file."
        )
    return key.strip()


_VALID_API_KEY: str = _load_api_key()


async def require_api_key(
    api_key: str | None = Security(API_KEY_HEADER),
) -> str:
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Supply it in the X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if not secrets.compare_digest(api_key.strip(), _VALID_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )

    return api_key
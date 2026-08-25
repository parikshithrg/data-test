"""Shared client for the public `api.mospi.gov.in` REST family (CPI, IIP,
WPI, National Accounts Statistics, PLFS, and RBI external-sector data) -
discovered live 2026-08-25 via the Swagger docs embedded in the eSankhyiki
portal (esankhyiki.mospi.gov.in), one `/layout/swagger_user_{product}.yaml`
file per product. Confirmed working for all 6 products used by
`macro_series.py`: `cpi`, `iip`, `wpi`, `nas`, `plfs`, `rbi`.

REAL, NOT A LOOPHOLE - every spec declares `security: bearerAuth`, but
none of it is enforced in practice: confirmed live via the portal's own
Swagger "Try it out" button, which calls the real endpoint with no
Authorization header at all and gets a 200 with real data. Treated here as
a genuine public API, not something to lean on quietly - if MoSPI starts
enforcing the declared auth, every fetch script using this module will
start failing loudly (401/403 from `raise_for_status`), not silently
degrade to stale/empty data.

WHY A CUSTOM SSL ADAPTER - api.mospi.gov.in's TLS stack rejects Python's
default OpenSSL 3.x "no unsafe legacy renegotiation" hardening (a
client-side policy, not a certificate problem): a plain `requests.get`
fails every time with `SSL: UNSAFE_LEGACY_RENEGOTIATION_DISABLED`. A
browser doesn't enable this check the same way, which is why the portal's
own Swagger UI works fine while a bare Python client doesn't.

PAGINATION - every endpoint uses the same `{data, meta_data: {page,
totalPages}, msg, statusCode}` envelope and caps `limit` at 100
(confirmed live: 500+ returns a 400 "Limit parameter too large"). `fetch_all`
walks every page at limit=100 and concatenates `data`.
"""

from __future__ import annotations

import ssl
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter

BASE_URL = "https://api.mospi.gov.in"
MAX_LIMIT = 100
_REQUEST_DELAY_SECONDS = 0.2  # polite pacing against a public government API


class _LegacyRenegotiationAdapter(HTTPAdapter):
    """Sets `ssl.OP_LEGACY_SERVER_CONNECT` so requests to api.mospi.gov.in
    don't fail under OpenSSL 3.x's stricter default renegotiation policy."""

    def init_poolmanager(self, *args: Any, **kwargs: Any) -> Any:
        ctx = ssl.create_default_context()
        ctx.options |= 0x4  # ssl.OP_LEGACY_SERVER_CONNECT (present on OpenSSL 3.x)
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)


def _session() -> requests.Session:
    s = requests.Session()
    s.mount("https://", _LegacyRenegotiationAdapter())
    return s


def fetch_all(path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    """GET every page of `path` (e.g. `/api/cpi/getCPIData`) with `params`,
    returning the concatenated `data` list. `params` must not include
    `limit`/`page` - both are managed here."""
    session = _session()
    rows: list[dict[str, Any]] = []
    page = 1
    total_pages = 1
    while page <= total_pages:
        r = session.get(
            f"{BASE_URL}{path}",
            params={**params, "limit": MAX_LIMIT, "page": page},
            headers={"accept": "*/*"},
            timeout=30,
        )
        r.raise_for_status()
        body = r.json()
        rows.extend(body.get("data") or [])
        total_pages = body.get("meta_data", {}).get("totalPages", 1)
        page += 1
        if page <= total_pages:
            time.sleep(_REQUEST_DELAY_SECONDS)
    return rows

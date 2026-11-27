"""HTTP connections to local inference services through SSH tunnels."""
import json
from http.client import IncompleteRead
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class InferenceHTTPError(OSError):
    def __init__(self, status):
        self.status = status
        super().__init__(f"inference endpoint returned HTTP {status}")


class InferenceUnavailable(OSError):
    """No complete inference response could be obtained through the tunnel."""


def post(url, payload, timeout=120):
    target = urlparse(url)
    if target.scheme not in ('http', 'https') or target.hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError("model URL must use a loopback SSH tunnel")
    opener = build_opener(ProxyHandler({}), _NoRedirect())
    request = Request(url, json.dumps(payload, ensure_ascii=False).encode(),
                      {"Content-Type": "application/json"})
    for attempt in range(2):
        try:
            with opener.open(request, timeout=timeout) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code in (429, 502, 503, 504):
                raise InferenceUnavailable(f"inference endpoint returned HTTP {error.code}") from error
            raise InferenceHTTPError(error.code) from error
        except URLError as error:
            raise InferenceUnavailable("inference endpoint is unreachable") from error
        except (IncompleteRead, ConnectionResetError, BrokenPipeError, TimeoutError) as error:
            if attempt:
                raise InferenceUnavailable("inference connection closed before a complete response") from error
            sleep(1)

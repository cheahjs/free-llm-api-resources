import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pull_available_models import fetch_gemini_limits


class DummyLogger:
    def info(self, message):
        pass

    def warning(self, message):
        pass

    def debug(self, message):
        pass


@patch("pull_available_models.cloudquotas_v1.CloudQuotasClient")
def test_fetch_gemini_limits_falls_back_when_credentials_missing(mock_client):
    mock_client.side_effect = Exception("credentials missing")

    result = fetch_gemini_limits(DummyLogger())

    assert result == {}

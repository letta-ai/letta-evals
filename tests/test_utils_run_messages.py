"""Unit tests for run-message pagination helper behavior."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from letta_evals.utils import list_all_run_messages


class _FakeRunsMessagesAPI:
    def __init__(self, pages: dict):
        self.pages = pages
        self.calls = []

    async def list(self, **kwargs):
        self.calls.append(kwargs)
        after = kwargs.get("after")
        return SimpleNamespace(items=self.pages.get(after, []))


class _FakeClient:
    def __init__(self, pages: dict):
        self.runs = SimpleNamespace(messages=_FakeRunsMessagesAPI(pages))


@pytest.mark.asyncio
async def test_list_all_run_messages_passes_params_and_dedupes_and_sorts():
    ts1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ts2 = datetime(2026, 1, 2, tzinfo=timezone.utc)
    ts3 = datetime(2026, 1, 3, tzinfo=timezone.utc)

    msg1 = SimpleNamespace(id="m1", created_at=ts1)
    msg2 = SimpleNamespace(id="m2", created_at=ts2)
    msg3 = SimpleNamespace(id="m3", created_at=ts3)

    # Intentionally out-of-order + duplicate across pages.
    pages = {
        None: [msg2, msg1],
        "m1": [msg1, msg3],
        "m3": [],
    }
    client = _FakeClient(pages)

    messages = await list_all_run_messages(client, "run-1", params={"return_token_ids": "true"})

    assert [m.id for m in messages] == ["m1", "m2", "m3"]
    assert client.runs.messages.calls[0]["run_id"] == "run-1"
    assert client.runs.messages.calls[0]["order"] == "asc"
    assert client.runs.messages.calls[0]["extra_query"] == {"return_token_ids": "true"}


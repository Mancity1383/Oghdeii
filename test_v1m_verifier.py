"""Unit tests for the v1m System-One voice guardrail.

Two modes are covered, both fully offline (no socket is ever opened):

* **online success** — a fake `typesafe_sdk` client returns a real
  ``SystemOneResponse``; the tests assert the request schema, the >= 0.80
  intent gate, the ``ignore`` veto, the risk gate, intent resolution and the
  exact-phrase cache.
* **offline fallback** — missing key, missing SDK, network errors, slow cloud
  (bounded wait) and a disabled toggle must all return a usable decision
  without raising and without blocking the caller's thread.

Run with ``pytest test_v1m_verifier.py`` or ``python -m unittest test_v1m_verifier``.
"""

import json
import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

from config_manager import ConfigManager
from oghdeii.voice import v1m_verifier as vm
from oghdeii.voice.v1m_verifier import (
    ACTIONS,
    V1MVoiceVerifier,
    normalize_phrase,
    sdk_available,
    sdk_base_url,
)

# typesafe_sdk is an optional runtime dependency: holding it in an Any-typed
# slot lets the "not installed" fallback assign None (mypy refuses to assign
# None to an imported class) while SDK_OK still gates every use of it.
SystemOneResponse: Any = None
try:
    from typesafe_sdk import SystemOneResponse
except Exception:  # pragma: no cover - SDK is an optional runtime dependency
    SystemOneResponse = None

SDK_OK = sdk_available() and SystemOneResponse is not None

ENV_KEYS = ("V1M_API_KEY", "TYPESAFE_API_KEY")


def make_response(probability=0.95, action="copy", risk=1.0):
    """A well-formed System-One answer body, decoded exactly like the SDK does."""
    answers = {
        "is_valid_command": {"type": "noul", "noul": probability},
        "action": {
            "type": "choice",
            "choice": action,
            "confidence": 0.91,
            "probabilities": {action: 0.91},
        },
    }
    if risk is not None:
        answers["execution_risk"] = {
            "type": "score",
            "score": risk,
            "confidence": 0.8,
            "legend": {"0": "safe", "4": "severe"},
            "probabilities": {"0": 0.7, "4": 0.3},
        }
    return SystemOneResponse.model_validate_json(json.dumps({
        "model": "v1m-latest",
        "usage": {"input_tokens": 96, "output_tokens": 21},
        "answers": answers,
    }))


class _FakeModels:
    def __init__(self, names=("v1m-latest",), error=None):
        self._names = list(names)
        self._error = error

    def list(self, timeout=None):
        if self._error is not None:
            raise self._error  # e.g. 401 from GET /v1/models
        return SimpleNamespace(
            models=[SimpleNamespace(name=name) for name in self._names]
        )


class _FakeClient:
    """Stands in for ``TypeSafeClient``: records the request, never opens a socket."""

    def __init__(self, response=None, error=None, delay=0.0):
        self._response = response   # built lazily: keep construction SDK-free
        self.error = error
        self.delay = delay
        self.calls = []
        self.closed = False
        self.models = _FakeModels(error=error)

    @property
    def response(self):
        if self._response is None:
            self._response = make_response()
        return self._response

    def system_one(self, *, state, questions, model, **_ignored):
        self.calls.append({"state": state, "questions": questions, "model": model})
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.response

    def close(self):
        self.closed = True


class VerifierTestBase(unittest.TestCase):
    """Shared fixture: temp config, scrubbed environment, pooled workers."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config = ConfigManager(self.tmp.name)
        self._saved_env = {key: os.environ.pop(key, None) for key in ENV_KEYS}
        self.factory_calls = []
        self.verifiers = []

    def tearDown(self):
        for verifier in self.verifiers:
            verifier.shutdown()
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.tmp.cleanup()

    def make_verifier(self, client=None, typing_probe=lambda: False, **kwargs):
        """Build a verifier whose "network" is a fake client factory."""
        client = _FakeClient() if client is None else client

        def factory(api_key, base_url, timeout_s):
            self.factory_calls.append((api_key, base_url, timeout_s))
            return client

        verifier = V1MVoiceVerifier(
            self.config, typing_probe=typing_probe, client_factory=factory, **kwargs
        )
        self.verifiers.append(verifier)
        self.last_client = client
        return verifier

    def enable(self, **overrides):
        values = {"enable_v1m_verification": True, "v1m_api_key": "test-key"}
        values.update(overrides)
        self.config.update(values)


@unittest.skipUnless(SDK_OK, "typesafe-sdk is not installed")
class OnlineVerificationTests(VerifierTestBase):
    """Online success mode: a cloud verdict drives allow/deny and resolution."""

    def test_online_success_allows_command(self):
        self.enable()
        verifier = self.make_verifier(_FakeClient(make_response(0.95, "copy", 1.0)))

        decision = verifier.verify("copy", "Laptop Copy", 0.93)

        self.assertEqual(decision.source, "cloud")
        self.assertTrue(decision.allow)
        self.assertEqual(decision.probability, 0.95)
        self.assertEqual(decision.canonical_action, "copy")
        self.assertEqual(decision.action_confidence, 0.91)
        self.assertEqual(decision.action_probability, 0.91)
        self.assertEqual(decision.command, "copy")
        self.assertEqual(decision.execution_risk, 1.0)
        self.assertEqual(verifier.stats["cloud"], 1)
        self.assertEqual(verifier.stats["blocked"], 0)

    def test_request_carries_the_required_single_pass_schema(self):
        self.enable()
        client = _FakeClient(make_response(0.9, "paste", 2.0))
        verifier = self.make_verifier(client)

        verifier.verify("paste", "Laptop Paste", 0.88)
        self.assertEqual(len(client.calls), 1)
        call = client.calls[0]

        # One pass, model mandated by the spec.
        self.assertEqual(call["model"], "v1m-latest")
        self.assertEqual(
            set(call["questions"]), {"is_valid_command", "action", "execution_risk"}
        )

        # Field types: Noul / Choice / Score.
        self.assertEqual(call["questions"]["is_valid_command"].type, "noul")
        self.assertEqual(call["questions"]["action"].type, "choice")
        self.assertEqual(call["questions"]["execution_risk"].type, "score")

        # Choice vocabulary is exactly the specified action set.
        self.assertEqual(tuple(call["questions"]["action"].criteria), ACTIONS)
        # The score rubric is an ordered, non-empty level list.
        rubric = call["questions"]["execution_risk"].criteria
        self.assertEqual(len(rubric), 5)

        # Context the model needs: phrase, recogniser read, typing state.
        state = call["state"]
        self.assertEqual(state["phrase"], "Laptop Paste")
        self.assertEqual(state["recognized_command"], "paste")
        self.assertIn("typing_active", state)
        self.assertIn("recognizer_confidence", state)

        exchange = json.loads(verifier.last_exchange_json())
        self.assertEqual(exchange["request"]["method"], "POST")
        self.assertEqual(
            exchange["request"]["url"], "https://v1m.ir/v1/systemone"
        )
        body = exchange["request"]["body"]
        self.assertEqual(body["state"], state)
        self.assertIn("show_desktop", body["questions"]["action"]["criteria"])
        self.assertEqual(exchange["parsed_verdict"]["action"], "paste")
        self.assertNotIn("test-key", json.dumps(exchange))

    def test_endpoint_and_credentials_are_passed_to_the_client(self):
        self.enable(v1m_endpoint="https://v1m.ir/v1")
        os.environ["V1M_API_KEY"] = "env-secret"
        verifier = self.make_verifier()

        verifier.verify("copy", "Laptop Copy", 0.9)

        api_key, base_url, timeout_s = self.factory_calls[0]
        self.assertEqual(api_key, "env-secret")   # env beats config.json
        self.assertEqual(base_url, "https://v1m.ir")  # SDK appends /v1/systemone
        self.assertGreater(timeout_s, 0)

    def test_low_intent_probability_is_blocked(self):
        self.enable()
        threshold = self.config.get("v1m_min_probability", 0.55)
        test_prob = round(threshold - 0.15, 2)
        verifier = self.make_verifier(_FakeClient(make_response(test_prob, "copy", 0.5)))

        decision = verifier.verify("copy", "laptop copy", 0.9)

        self.assertFalse(decision.allow)
        self.assertIn(f"{test_prob:.2f}", decision.reason)
        self.assertIn(f"{threshold:.2f}", decision.reason)
        self.assertEqual(verifier.stats["blocked"], 1)

    def test_rejection_reason_dynamically_reflects_custom_threshold(self):
        self.enable(v1m_min_probability=0.75)
        verifier = self.make_verifier(_FakeClient(make_response(0.60, "copy", 0.5)))

        decision = verifier.verify("copy", "laptop copy", 0.9)

        self.assertFalse(decision.allow)
        self.assertIn("is_valid_command 0.60 < 0.75", decision.reason)

    def test_ignore_choice_vetoes_execution(self):
        self.enable()
        verifier = self.make_verifier(_FakeClient(make_response(0.97, "ignore", 0.5)))

        decision = verifier.verify("copy", "Laptop Copy", 0.95)

        self.assertFalse(decision.allow)
        self.assertIn("'ignore'", decision.reason)
        self.assertEqual(decision.command, "copy")  # untouched, but never runs

    def test_execution_risk_gate_blocks_when_tightened(self):
        self.enable(v1m_max_execution_risk=2.0)
        verifier = self.make_verifier(_FakeClient(make_response(0.95, "copy", 3.5)))

        decision = verifier.verify("copy", "Laptop Copy", 0.9)

        self.assertFalse(decision.allow)
        self.assertIn("execution_risk 3.5", decision.reason)

    def test_execution_risk_gate_is_open_by_default(self):
        self.enable()  # v1m_max_execution_risk defaults to 4.0
        verifier = self.make_verifier(_FakeClient(make_response(0.95, "copy", 3.9)))

        self.assertTrue(verifier.verify("copy", "Laptop Copy", 0.9).allow)

    def test_missing_risk_answer_is_conservative_under_an_active_gate(self):
        self.enable(v1m_max_execution_risk=1.0)
        verifier = self.make_verifier(_FakeClient(make_response(0.95, "copy", None)))

        decision = verifier.verify("copy", "Laptop Copy", 0.9)

        self.assertFalse(decision.allow)
        self.assertIn("execution_risk unavailable", decision.reason)

    def test_model_overrides_a_misrecognised_local_command(self):
        self.enable()
        # Recogniser said "paste", but the model resolved the phrase to "copy".
        verifier = self.make_verifier(_FakeClient(make_response(0.96, "copy", 0.5)))

        decision = verifier.verify("paste", "Laptop Copy", 0.9)

        self.assertTrue(decision.allow)
        self.assertEqual(decision.command, "copy")

    def test_free_transcript_lets_v1m_choose_the_action(self):
        self.enable()
        client = _FakeClient(make_response(0.94, "copy", 0.5))
        verifier = self.make_verifier(client)
        alternates = [
            {"text": "lot of hockey", "confidence": 0.31},
            {"text": "Laptop Copy", "confidence": 0.72},
        ]

        decision = verifier.verify("", "lot of hockey", 0.31, alternates=alternates)

        self.assertTrue(decision.allow)
        self.assertEqual(decision.command, "copy")
        self.assertIsNone(client.calls[0]["state"]["recognized_command"])
        self.assertEqual(client.calls[0]["state"]["phrase"], "lot of hockey")
        self.assertEqual(
            client.calls[0]["state"]["transcript_candidates"],
            alternates,
        )
        valid_command_question = client.calls[0]["questions"]["is_valid_command"]
        self.assertIn("transcript_candidates", str(valid_command_question).lower())

    def test_free_transcript_fails_closed_if_v1m_cannot_answer(self):
        self.enable(v1m_wait_timeout_ms=60)
        client = _FakeClient(error=ConnectionError("network unavailable"))
        verifier = self.make_verifier(client)

        decision = verifier.verify("", "mamad desktop", 0.88)

        self.assertFalse(decision.allow)
        self.assertEqual(decision.command, "")
        self.assertIn("not executed without a V1M decision", decision.reason)

    def test_constrained_recognition_is_sent_as_a_command_hint(self):
        self.enable()
        client = _FakeClient(make_response(0.96, "copy", 1.0))
        verifier = self.make_verifier(client)

        decision = verifier.verify(
            "copy",
            "laptop copy",
            0.756,
            alternates=[
                {"text": "laptop copy", "confidence": 0.756},
                {"text": "lap top copy", "confidence": 0.658},
            ],
        )

        self.assertTrue(decision.allow)
        state = client.calls[0]["state"]
        self.assertEqual(state["recognized_command"], "copy")
        self.assertEqual(state["transcript_candidates"][1]["text"], "lap top copy")

    def test_low_intent_rejection_preserves_choice_and_risk_evidence(self):
        self.enable(v1m_min_probability=0.30)
        client = _FakeClient(make_response(0.20, "copy", 1.62))
        verifier = self.make_verifier(client)

        decision = verifier.verify("copy", "laptop copy", 0.756)

        self.assertFalse(decision.allow)
        self.assertEqual(decision.action_confidence, 0.91)
        self.assertIn("is_valid_command 0.20 < 0.30", decision.reason)
        self.assertIn("action=copy", decision.reason)
        self.assertIn("Choice confidence 91%", decision.reason)
        exchange = json.loads(verifier.last_exchange_json())
        self.assertEqual(exchange["parsed_verdict"]["action_probabilities"], {"copy": 0.91})

    def test_extended_model_vocabulary_covers_undo(self):
        self.enable()
        # The cloud Choice schema includes every supported voice command.
        verifier = self.make_verifier(_FakeClient(make_response(0.94, "undo", 0.5)))

        decision = verifier.verify("undo", "Laptop Undo That", 0.91)

        self.assertTrue(decision.allow)
        self.assertEqual(decision.command, "undo")

    def test_play_and_pause_map_onto_the_play_pause_action(self):
        self.enable()
        verifier = self.make_verifier(_FakeClient(make_response(0.99, "play_pause", 0.2)))

        decision = verifier.verify("pause", "Laptop Pause", 0.95)

        self.assertTrue(decision.allow)
        self.assertEqual(decision.canonical_action, "play_pause")
        self.assertEqual(decision.command, "pause")  # local alias preserved

    def test_malformed_action_falls_back_offline_instead_of_raising(self):
        self.enable()
        verifier = self.make_verifier(
            _FakeClient(make_response(0.9, "banana", 0.5))
        )

        decision = verifier.verify("copy", "Laptop Copy", 0.9)

        self.assertEqual(decision.source, "offline")
        self.assertFalse(decision.allow)
        self.assertIn("unknown action", verifier.last_error)


@unittest.skipUnless(SDK_OK, "typesafe-sdk is not installed")
class CachingAndLatencyTests(VerifierTestBase):
    """Bounded waits, late-cache warm-up and exact-phrase cache hits."""

    def test_identical_phrase_bypasses_remote_evaluation(self):
        self.enable()
        client = _FakeClient(make_response(0.95, "copy", 1.0))
        verifier = self.make_verifier(client)

        first = verifier.verify("copy", "Laptop Copy", 0.9)
        second = verifier.verify("copy", "Laptop Copy!", 0.9)   # punctuation variant
        third = verifier.verify("copy", "  LAPTOP   copy ", 0.9)  # spacing/case

        self.assertEqual(first.source, "cloud")
        self.assertEqual(second.source, "cache")
        self.assertEqual(third.source, "cache")
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(verifier.stats["cache"], 2)
        self.assertTrue(second.allow)
        self.assertEqual(second.probability, first.probability)

    def test_cache_entry_expires_after_the_configured_ttl(self):
        self.enable(v1m_cache_ttl_s=0)  # TTL 0 disables caching entirely
        client = _FakeClient(make_response(0.95, "copy", 1.0))
        verifier = self.make_verifier(client)

        verifier.verify("copy", "Laptop Copy", 0.9)
        verifier.verify("copy", "Laptop Copy", 0.9)

        self.assertEqual(len(client.calls), 2)

    def test_cached_verdict_is_re_evaluated_against_live_thresholds(self):
        self.enable()
        client = _FakeClient(make_response(0.85, "copy", 1.0))
        verifier = self.make_verifier(client)

        self.assertTrue(verifier.verify("copy", "Laptop Copy", 0.9).allow)
        self.config.set("v1m_min_probability", 0.90)  # tighten without restart

        decision = verifier.verify("copy", "Laptop Copy", 0.9)
        self.assertEqual(decision.source, "cache")
        self.assertFalse(decision.allow)
        self.assertEqual(len(client.calls), 1)

    def test_slow_cloud_returns_offline_within_the_wait_budget(self):
        self.enable(v1m_wait_timeout_ms=60)
        client = _FakeClient(make_response(0.95, "copy", 1.0), delay=0.4)
        verifier = self.make_verifier(client)

        started = time.monotonic()
        decision = verifier.verify("copy", "Laptop Copy", 0.9)
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.30, "caller thread was blocked by the network")
        self.assertEqual(decision.source, "offline")
        self.assertFalse(decision.allow)
        self.assertIn("60 ms", decision.reason)
        self.assertIn("not executed without a V1M decision", decision.reason)
        self.assertEqual(verifier.stats["timeouts"], 1)

        # The late answer must warm the cache, not be thrown away.
        time.sleep(0.6)
        replay = verifier.verify("copy", "Laptop Copy", 0.9)
        self.assertEqual(replay.source, "cache")
        self.assertEqual(replay.probability, 0.95)
        self.assertEqual(len(client.calls), 1)

    def test_typing_state_is_sent_and_splits_the_cache(self):
        self.enable()
        client = _FakeClient(make_response(0.95, "lock", 3.0))
        typing_active = [False]
        verifier = self.make_verifier(
            client, typing_probe=lambda: typing_active[0]
        )

        verifier.verify("lock", "Laptop Lock", 0.9)
        typing_active[0] = True
        verifier.verify("lock", "Laptop Lock", 0.9)

        self.assertEqual(len(client.calls), 2)
        states = [call["state"]["typing_active"] for call in client.calls]
        self.assertEqual(states, [False, True])

    def test_cache_splits_when_asr_confidence_or_wake_policy_changes(self):
        self.enable()
        client = _FakeClient(make_response(0.95, "copy", 1.0))
        verifier = self.make_verifier(client)

        verifier.verify("copy", "Laptop Copy", 0.90)
        verifier.verify("copy", "Laptop Copy", 0.72)
        self.config.set("voice_require_wake_word", False)
        verifier.verify("copy", "Laptop Copy", 0.72)

        self.assertEqual(len(client.calls), 3)

    def test_verify_async_resolves_without_blocking_the_caller(self):
        self.enable()
        verifier = self.make_verifier(_FakeClient(make_response(0.93, "mute", 0.4)))

        future = verifier.verify_async("mute", "Laptop Mute", 0.9)
        decision = future.result(timeout=3.0)

        self.assertTrue(decision.allow)
        self.assertEqual(decision.command, "mute")


@unittest.skipUnless(SDK_OK, "typesafe-sdk is not installed")
class OfflineFallbackTests(VerifierTestBase):
    """V1M failures fail closed; explicitly disabled V1M keeps local mode."""

    def test_disabled_toggle_never_touches_credentials_or_network(self):
        # enable_v1m_verification defaults to False in DEFAULT_CONFIG.
        verifier = self.make_verifier()

        decision = verifier.verify("copy", "Laptop Copy", 0.9)

        self.assertEqual(decision.source, "disabled")
        self.assertTrue(decision.allow)
        self.assertIsNone(decision.probability)
        self.assertEqual(decision.command, "copy")
        self.assertEqual(self.factory_calls, [])

    def test_missing_key_falls_back_offline_without_building_a_client(self):
        self.enable(v1m_api_key="")  # enabled, but no key anywhere
        verifier = self.make_verifier()

        decision = verifier.verify("copy", "Laptop Copy", 0.9)

        self.assertEqual(decision.source, "offline")
        self.assertFalse(decision.allow)
        self.assertIsNone(decision.probability)
        self.assertIn("no V1M_API_KEY", decision.reason)
        self.assertEqual(self.factory_calls, [])
        self.assertEqual(verifier.stats["offline"], 1)

    def test_network_error_falls_back_offline_without_raising(self):
        self.enable()
        client = _FakeClient(error=ConnectionError("DNS lookup failed"))
        verifier = self.make_verifier(client)

        decision = verifier.verify("copy", "Laptop Copy", 0.9)

        self.assertEqual(decision.source, "offline")
        self.assertFalse(decision.allow)
        self.assertIn("not executed without a V1M decision", decision.reason)
        self.assertIn("DNS lookup failed", verifier.last_error)

    def test_repeated_failures_open_a_circuit_then_pause_the_cloud(self):
        self.enable()
        client = _FakeClient(error=OSError("connection refused"))
        verifier = self.make_verifier(client)

        for _ in range(3):
            self.assertEqual(verifier.verify("copy", "Laptop Copy", 0.9).source, "offline")
        self.assertEqual(len(client.calls), 3)   # one attempt per failure

        paused = verifier.verify("copy", "Laptop Copy", 0.9)
        self.assertEqual(paused.source, "offline")
        self.assertIn("cloud paused", paused.reason)
        self.assertEqual(len(client.calls), 3)   # circuit is holding calls back

    def test_verify_never_raises_even_with_a_broken_client(self):
        self.enable()

        class ExplodingClient:
            models = _FakeModels()

            def system_one(self, **_kwargs):
                raise RuntimeError("everything is broken")

            def close(self):
                pass

        verifier = self.make_verifier(ExplodingClient())
        decision = verifier.verify("copy", "Laptop Copy", 0.9)
        self.assertEqual(decision.source, "offline")
        self.assertFalse(decision.allow)

    def test_missing_sdk_is_reported_and_falls_back_offline(self):
        self.enable()
        verifier = self.make_verifier()

        with mock.patch.object(vm, "_typesafe_sdk", None):
            self.assertFalse(sdk_available())
            decision = verifier.verify("copy", "Laptop Copy", 0.9)

        self.assertEqual(decision.source, "offline")
        self.assertFalse(decision.allow)
        self.assertIn("typesafe-sdk is not installed", decision.reason)
        self.assertEqual(self.factory_calls, [])

    def test_describe_reports_status_without_secrets(self):
        self.enable()
        verifier = self.make_verifier()

        info = verifier.describe()

        self.assertTrue(info["enabled"])
        self.assertEqual(info["endpoint"], "https://v1m.ir/v1")
        self.assertEqual(info["model"], "v1m-latest")
        self.assertEqual(info["key_source"], "app configuration")
        self.assertNotIn("test-key", json.dumps(info))


class ConnectionTestTests(VerifierTestBase):
    """The Settings "Test Connection" button path."""

    def test_reports_a_missing_key_without_touching_the_network(self):
        self.config.set("v1m_api_key", "")
        verifier = self.make_verifier()

        ok, message = verifier.test_connection()

        self.assertFalse(ok)
        self.assertIn("No API key", message)
        self.assertEqual(self.factory_calls, [])

    @unittest.skipUnless(SDK_OK, "typesafe-sdk is not installed")
    def test_reports_a_successful_probe(self):
        self.enable()
        verifier = self.make_verifier(_FakeClient())

        ok, message = verifier.test_connection()

        self.assertTrue(ok)
        self.assertIn("v1m-latest", message)
        self.assertEqual(len(self.factory_calls), 1)

    @unittest.skipUnless(SDK_OK, "typesafe-sdk is not installed")
    def test_reports_an_authentication_failure_in_plain_language(self):
        self.enable()
        if not SDK_OK:  # pragma: no cover - guarded by the decorator
            return
        import httpx2
        from typesafe_sdk import TypeSafeAuthenticationError

        error = TypeSafeAuthenticationError(
            401, {"error": "invalid key"}, httpx2.Headers(), endpoint="POST /v1/systemone"
        )
        verifier = self.make_verifier(_FakeClient(error=error))

        ok, message = verifier.test_connection()

        self.assertFalse(ok)
        self.assertIn("rejected the API key", message)


class UtilityTests(VerifierTestBase):
    """SDK-independent behaviour: endpoint normalisation, keys, config."""

    def test_endpoint_is_shortened_so_the_sdk_does_not_double_the_version(self):
        self.assertEqual(sdk_base_url("https://v1m.ir/v1"), "https://v1m.ir")
        self.assertEqual(sdk_base_url("https://v1m.ir/v1/"), "https://v1m.ir")
        self.assertEqual(sdk_base_url("https://v1m.ir"), "https://v1m.ir")
        self.assertEqual(sdk_base_url(None), "https://v1m.ir")
        self.assertEqual(sdk_base_url("https://api.example.com"), "https://api.example.com")

    def test_phrase_normalisation_folds_case_punctuation_and_spacing(self):
        self.assertEqual(normalize_phrase("  Laptop, Copy! "), "laptop copy")
        self.assertEqual(normalize_phrase("LAPTOP\tcopy"), "laptop copy")
        self.assertEqual(normalize_phrase(None), "")

    def test_disabled_by_default_and_verify_is_inert(self):
        verifier = self.make_verifier()

        decision = verifier.verify("copy", "Laptop Copy", 0.9)

        self.assertFalse(verifier.enabled)
        self.assertEqual(decision.source, "disabled")
        self.assertTrue(decision.allow)
        self.assertEqual(self.factory_calls, [])

    def test_verify_returns_offline_result_when_shut_down(self):
        self.enable()
        verifier = self.make_verifier()
        verifier.shutdown()

        decision = verifier.verify("copy", "Laptop Copy", 0.9)
        self.assertEqual(decision.source, "offline")
        self.assertFalse(decision.allow)

    def test_v1m_defaults_are_declared_and_sanitised(self):
        from config_manager import DEFAULT_CONFIG

        for key in (
            "enable_v1m_verification", "v1m_api_key", "v1m_endpoint", "v1m_model",
            "v1m_wait_timeout_ms", "v1m_request_timeout_ms", "v1m_min_probability",
            "v1m_max_execution_risk", "v1m_cache_ttl_s", "v1m_failure_cooldown_s",
        ):
            self.assertIn(key, DEFAULT_CONFIG)
        self.assertFalse(DEFAULT_CONFIG["enable_v1m_verification"])
        self.assertEqual(DEFAULT_CONFIG["v1m_wait_timeout_ms"], 3000)
        self.assertEqual(DEFAULT_CONFIG["v1m_min_probability"], 0.55)

        # Hostile/garbage values must never reach the client constructor.
        self.config.update({
            "v1m_wait_timeout_ms": "not-a-number",
            "v1m_min_probability": 99,
            "v1m_max_execution_risk": -5,
            "v1m_endpoint": "",
            "v1m_model": None,
        })
        self.assertEqual(self.config.get("v1m_wait_timeout_ms"), 3000)
        self.assertEqual(self.config.get("v1m_min_probability"), 1.0)
        self.assertEqual(self.config.get("v1m_max_execution_risk"), 0.0)
        self.assertEqual(self.config.get("v1m_endpoint"), "https://v1m.ir/v1")
        self.assertEqual(self.config.get("v1m_model"), "v1m-latest")

    def test_api_key_prefers_environment_over_config(self):
        verifier = self.make_verifier()
        self.config.set("v1m_api_key", "from-config")

        self.assertEqual(verifier.api_key(), "from-config")
        os.environ["V1M_API_KEY"] = "from-env"
        self.assertEqual(verifier.api_key(), "from-env")
        self.assertEqual(verifier.key_source(), "environment (V1M_API_KEY)")


if __name__ == "__main__":
    unittest.main(verbosity=2)

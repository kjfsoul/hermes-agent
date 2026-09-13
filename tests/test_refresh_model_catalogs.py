"""Tests for ~/.hermes/scripts/refresh-model-catalogs.py (v3.4).

Covers the required-provider contract (direct fetch, failure classification,
last-known-good preservation), both-direction drift audit, picker authority
(live authoritative / LKG-cache fallback / static bootstrap), and config
non-regression. All network and filesystem effects are mocked; deterministic.

NOTE: tests/conftest.py sandboxes HERMES_HOME to a throwaway tempdir. The
refresh script under test lives in the REAL Hermes home, so it is loaded by
path with the real home injected for the duration of its module exec, and
hermes_constants.get_hermes_home is patched (not reloaded) wherever the
cache layer needs a writable sandbox.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

_REAL_HERMES_HOME = os.path.expanduser("~/.hermes")
_SCRIPT = os.path.join(_REAL_HERMES_HOME, "scripts", "refresh-model-catalogs.py")
_spec = importlib.util.spec_from_file_location("refresh_model_catalogs", _SCRIPT)


def _load_script():
    if "refresh_model_catalogs" in sys.modules:
        return sys.modules["refresh_model_catalogs"]
    import hermes_cli.models as hm  # noqa: F401  (script imports this too)

    mod = importlib.util.module_from_spec(_spec)
    sys.modules["refresh_model_catalogs"] = mod
    with mock.patch.dict(os.environ, {"HERMES_HOME": _REAL_HERMES_HOME}):
        mod.__loader__.exec_module(mod)
    return mod


R = _load_script()

REQUIRED = ["kilocode", "openrouter", "nous", "zai", "deepseek"]


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://example.test/models", code=code, msg="x",
        hdrs=None, fp=None,
    )


class _Profile:
    """Stand-in for providers.base.ProviderProfile."""

    def __init__(self, behavior):
        self._behavior = behavior
        self.name = "test"
        self.base_url = "https://example.test/v1"
        self.fallback_models = ()
        self.auth_type = "api_key"

    def fetch_models(self, *, api_key=None, base_url=None, timeout=8.0):
        if callable(self._behavior):
            return self._behavior(api_key=api_key, base_url=base_url)
        return self._behavior


def _raises(exc):
    def _inner(api_key=None, base_url=None):
        raise exc
    return _inner


def _patch_direct_fetch(behavior):
    """Patch the SOURCE modules the script imports lazily inside
    fetch_required_provider (function-level imports bypass module attrs)."""
    return [
        mock.patch("providers.get_provider_profile", return_value=_Profile(behavior)),
        mock.patch("hermes_cli.auth.resolve_api_key_provider_credentials",
                   create=True,
                   return_value={"api_key": "k", "base_url": "https://x"}),
        mock.patch("hermes_cli.models.update_provider_cache_entry"),
    ]


class RequiredProviderSuccess(unittest.TestCase):
    """Per-provider success path + cache update on success only."""

    def test_each_required_provider_success_updates_cache(self):
        import hermes_cli.models as hm

        for slug in ["zai", "deepseek", "kilocode"]:
            with self.subTest(slug=slug):
                patches = _patch_direct_fetch(["model-a", "model-b"])
                with patches[0], patches[1], patches[2] as upce:
                    ok, models, detail = R.fetch_required_provider(slug)
                self.assertTrue(ok, detail)
                self.assertEqual(models, ["model-a", "model-b"])
                upce.assert_called_once_with(slug, ["model-a", "model-b"])

    def test_openrouter_public_catalog_success(self):
        ok_payload = {"data": [{"id": "vendor/m1"}, {"id": "vendor/m2"}]}
        resp = mock.MagicMock()
        resp.status = 200
        resp.read.return_value = json.dumps(ok_payload).encode()
        ctx = mock.MagicMock()
        ctx.__enter__ = mock.Mock(return_value=resp)
        ctx.__exit__ = mock.Mock(return_value=False)
        with mock.patch("urllib.request.urlopen", return_value=ctx):
            ok, models, detail = R.fetch_required_provider("openrouter")
        self.assertTrue(ok, detail)
        self.assertEqual(len(models), 2)

    def test_openrouter_http_error_classified(self):
        with mock.patch("urllib.request.urlopen",
                        side_effect=_http_error(429)):
            ok, models, detail = R.fetch_required_provider("openrouter")
        self.assertFalse(ok)
        self.assertIn("http_429", detail)

    def test_openrouter_malformed_payload_classified(self):
        resp = mock.MagicMock()
        resp.status = 200
        resp.read.return_value = b"<html>not json</html>"
        ctx = mock.MagicMock()
        ctx.__enter__ = mock.Mock(return_value=resp)
        ctx.__exit__ = mock.Mock(return_value=False)
        with mock.patch("urllib.request.urlopen", return_value=ctx):
            ok, models, detail = R.fetch_required_provider("openrouter")
        self.assertFalse(ok)
        self.assertIn("malformed_json", detail)

    def test_openrouter_empty_catalog_classified(self):
        resp = mock.MagicMock()
        resp.status = 200
        resp.read.return_value = json.dumps({"data": []}).encode()
        ctx = mock.MagicMock()
        ctx.__enter__ = mock.Mock(return_value=resp)
        ctx.__exit__ = mock.Mock(return_value=False)
        with mock.patch("urllib.request.urlopen", return_value=ctx):
            ok, models, detail = R.fetch_required_provider("openrouter")
        self.assertFalse(ok)
        self.assertIn("unexpected empty catalog", detail)

    def test_nous_success(self):
        with mock.patch("hermes_cli.auth.resolve_nous_runtime_credentials",
                        create=True,
                        return_value={"api_key": "k", "base_url": "https://x"}), \
             mock.patch("hermes_cli.auth.fetch_nous_models",
                        create=True,
                        return_value=["nous/m1", "nous/m2"]), \
             mock.patch("hermes_cli.models.update_provider_cache_entry") as upce:
            ok, models, detail = R.fetch_required_provider("nous")
        self.assertTrue(ok, detail)
        self.assertEqual(models, ["nous/m1", "nous/m2"])
        upce.assert_called_once_with("nous", ["nous/m1", "nous/m2"])

    def test_nous_empty_catalog_is_failure(self):
        with mock.patch("hermes_cli.auth.resolve_nous_runtime_credentials",
                        create=True,
                        return_value={"api_key": "k", "base_url": "https://x"}), \
             mock.patch("hermes_cli.auth.fetch_nous_models",
                        create=True, return_value=[]):
            ok, models, detail = R.fetch_required_provider("nous")
        self.assertFalse(ok)
        self.assertIn("unexpected empty catalog", detail)

    def test_main_returns_zero_when_all_required_pass(self):
        def fake_fetch(slug):
            if slug == "nous":
                return True, ["n1"], "live portal fetch"
            return True, [f"{slug}-1"], "ok"

        patches = [
            mock.patch.object(R, "fetch_required_provider", side_effect=fake_fetch),
            mock.patch.object(R, "discover_openrouter_free", return_value={
                "free_models": [], "free_blocked": [], "paid_top": [],
                "total_catalog": 0}),
            mock.patch.object(R, "discover_kilocode_free", return_value={
                "free_models": [], "auto_routers": [], "free_count": 0,
                "total_models": 0}),
            mock.patch.object(R, "discover_nous_free", return_value={
                "free_models": [], "free_vision_model": None, "source": "live"}),
            mock.patch.object(R, "update_catalog_cache"),
            mock.patch.object(R, "save_free_report"),
            mock.patch.object(R, "fetch_models_dev", return_value={}),
            mock.patch.object(R, "cached_provider_model_ids", return_value=[]),
            mock.patch.object(R, "audit_stale_drift"),
            mock.patch.object(R, "log", lambda *a, **k: None),
        ]
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patches[5], patches[6], patches[7], patches[8], patches[9]:
            rc = R.main()
        self.assertEqual(rc, 0)


class RequiredProviderFailure(unittest.TestCase):
    """Failure classification; cache untouched; nonzero exit."""

    def _run(self, behavior, expect_frag):
        patches = _patch_direct_fetch(behavior)
        with patches[0], patches[1], patches[2] as upce:
            ok, models, detail = R.fetch_required_provider("zai")
        self.assertFalse(ok)
        self.assertEqual(models, [])
        self.assertIn(expect_frag, detail)
        upce.assert_not_called()  # last-known-good cache preserved

    def test_401(self):
        self._run(_raises(_http_error(401)), "http_401")

    def test_403(self):
        self._run(_raises(_http_error(403)), "http_403")

    def test_429(self):
        self._run(_raises(_http_error(429)), "http_429")

    def test_timeout(self):
        self._run(
            _raises(urllib.error.URLError(TimeoutError("timed out"))),
            "timeout",
        )

    def test_malformed_json_raises_is_classified(self):
        self._run(
            _raises(json.JSONDecodeError("Expecting value", "<html>", 0)),
            "malformed_json",
        )

    def test_none_return_is_failure(self):
        self._run(None, "fetch_models returned None")

    def test_empty_catalog_is_failure(self):
        self._run([], "unexpected empty catalog")

    def test_nous_credential_failure_is_classified(self):
        import hermes_cli.auth as ha

        with mock.patch.object(
            ha, "resolve_nous_runtime_credentials",
            side_effect=ha.AuthError("not logged in", provider="nous"),
        ):
            ok, models, detail = R.fetch_required_provider("nous")
        self.assertFalse(ok)
        self.assertIn("credential resolution failed", detail)

    def test_main_exits_nonzero_on_required_failure(self):
        def fake_fetch(slug):
            if slug == "deepseek":
                return False, [], "http_401"
            if slug == "nous":
                return True, ["n1"], "live portal fetch"
            return True, [f"{slug}-1"], "ok"

        patches = [
            mock.patch.object(R, "fetch_required_provider", side_effect=fake_fetch),
            mock.patch.object(R, "discover_openrouter_free", return_value={
                "free_models": [], "free_blocked": [], "paid_top": [],
                "total_catalog": 0}),
            mock.patch.object(R, "discover_kilocode_free", return_value={
                "free_models": [], "auto_routers": [], "free_count": 0,
                "total_models": 0}),
            mock.patch.object(R, "discover_nous_free", return_value={
                "free_models": [], "free_vision_model": None, "source": "live"}),
            mock.patch.object(R, "update_catalog_cache"),
            mock.patch.object(R, "save_free_report"),
            mock.patch.object(R, "fetch_models_dev", return_value={}),
            mock.patch.object(R, "cached_provider_model_ids", return_value=[]),
            mock.patch.object(R, "audit_stale_drift"),
            mock.patch.object(R, "log", lambda *a, **k: None),
        ]
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patches[5], patches[6], patches[7], patches[8], patches[9]:
            rc = R.main()
        self.assertEqual(rc, 1)


class DriftAudit(unittest.TestCase):
    """Both-direction drift semantics."""

    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        src = os.path.join(self.tmp.name, "hermes_cli")
        os.makedirs(src)
        with open(os.path.join(src, "models.py"), "w") as f:
            f.write(
                'catalog: dict[str, list[str]] = {\n'
                '"zai": [\n'
                '"glm-5.3",\n'
                '"glm-stale",\n'
                '],\n'
                '"kilocode": [\n'
                '"kilo-pick-1",\n'
                '"kilo-dead-pick",\n'
                '],\n'
                '"groq": [\n'
                '"g-1",\n'
                '],\n'
                '}\n'
            )
        for attr, value in (
            ("HERMES_SOURCE", self.tmp.name),
            ("DRIFT_REPORT_PATH", os.path.join(self.tmp.name, "drift.json")),
        ):
            p = mock.patch.object(R, attr, value)
            p.start()
            self.addCleanup(p.stop)
        quiet = mock.patch.object(R, "log", lambda *a, **k: None)
        quiet.start()
        self.addCleanup(quiet.stop)

    def _audit(self, live):
        R.audit_stale_drift(live)
        with open(os.path.join(self.tmp.name, "drift.json")) as f:
            return json.load(f)

    def test_stale_and_missing_both_directions_required(self):
        report = self._audit({"zai": ["glm-5.3", "glm-new"], "groq": ["g-1"]})
        zai = report["providers"]["zai"]
        self.assertEqual(zai["stale_static"], ["glm-stale"])
        self.assertEqual(zai["missing_static"], ["glm-new"])
        self.assertTrue(report["any_drift"])
        self.assertEqual(zai["missing_static_scope"], "drift")

    def test_match_requires_both_empty(self):
        report = self._audit({"zai": ["glm-5.3", "glm-stale"], "groq": ["g-1"]})
        zai = report["providers"]["zai"]
        self.assertEqual(zai["stale_static"], [])
        self.assertEqual(zai["missing_static"], [])
        self.assertFalse(report["any_drift"])

    def test_missing_static_informational_for_non_required(self):
        # groq live serves a model the curated subset omits: recorded, but
        # NOT drift (curated subsets are intentional for non-required).
        report = self._audit({"zai": ["glm-5.3", "glm-stale"], "groq": ["g-1", "g-new"]})
        groq = report["providers"]["groq"]
        self.assertEqual(groq["missing_static"], ["g-new"])
        self.assertEqual(groq["missing_static_scope"], "informational-non-required")
        self.assertFalse(report["any_drift"])

    def test_missing_static_alone_flips_drift_for_required(self):
        # No stale entries anywhere — a live-only model on a REQUIRED
        # provider is still drift (both-direction rule).
        report = self._audit({"zai": ["glm-5.3", "glm-stale", "z-new"], "groq": ["g-1"]})
        self.assertTrue(report["any_drift"])
        self.assertEqual(report["providers"]["zai"]["stale_static"], [])
        self.assertEqual(report["providers"]["zai"]["missing_static"], ["z-new"])


class PickerAuthority(unittest.TestCase):
    """Live authoritative / LKG cache / static bootstrap.

    Uses hermes_constants.get_hermes_home patching (source of truth for the
    cache path); hermes_cli.models is imported fresh per test WITHOUT reload.
    """

    def setUp(self):
        import tempfile

        import hermes_constants as hc

        self.hm = tempfile.TemporaryDirectory()
        p = mock.patch.object(hc, "get_hermes_home", return_value=Path(self.hm.name))
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(self.hm.cleanup)

    def _cache_entry(self, models, at):
        path = os.path.join(self.hm.name, "provider_models_cache.json")
        with open(path, "w") as f:
            json.dump({"zai": {"fp": "x", "at": at, "models": models}}, f)

    def test_expired_cache_served_stale_while_revalidate(self):
        import hermes_cli.models as hm

        # 1 day ago: past the 1h TTL, inside the 7-day stale-serve window
        self._cache_entry(["old-1"], at=hm.time.time() - 86400)
        with mock.patch.object(hm, "_credential_fingerprint", return_value="x"), \
             mock.patch.object(hm, "_spawn_swr_refresh") as swr:
            got = hm.cached_provider_model_ids("zai")
        self.assertEqual(got, ["old-1"])  # LKG served
        swr.assert_called_once_with("zai")  # refresh spawned off-thread

    def test_beyond_stale_window_refetches(self):
        import hermes_cli.models as hm

        # 8 days ago: past even the 7-day stale-serve window — LKG must NOT
        # be served indefinitely; a live refetch attempt happens instead.
        self._cache_entry(["ancient-1"], at=hm.time.time() - 8 * 86400)
        with mock.patch.object(hm, "_credential_fingerprint", return_value="x"), \
             mock.patch.object(hm, "provider_model_ids",
                               return_value=["refetched-1"]):
            got = hm.cached_provider_model_ids("zai")
        self.assertEqual(got, ["refetched-1"])

    def test_fresh_cache_served_direct(self):
        import hermes_cli.models as hm

        self._cache_entry(["fresh-1"], at=hm.time.time())
        with mock.patch.object(hm, "_credential_fingerprint", return_value="x"):
            got = hm.cached_provider_model_ids("zai")
        self.assertEqual(got, ["fresh-1"])

    def test_static_bootstrap_when_no_cache_and_no_live(self):
        import hermes_cli.models as hm

        # provider_model_ids falls through to the curated static list when
        # there are no credentials and no live result (bootstrap path).
        profile = _Profile(None)
        with mock.patch("providers.get_provider_profile", return_value=profile), \
             mock.patch("hermes_cli.auth.resolve_api_key_provider_credentials",
                        create=True,
                        return_value={"api_key": "", "base_url": ""}), \
             mock.patch.object(hm, "_merge_with_models_dev",
                               side_effect=lambda p, c: list(c)):
            got = hm.provider_model_ids("zai")
        self.assertEqual(got, list(hm._PROVIDER_MODELS["zai"]))
        self.assertNotIn("glm-5v-turbo", got)

    def test_picker_excludes_stale_static_after_live_refresh(self):
        """Successful live fetch = authoritative picker list for the five
        required providers; curated-only entries must NOT be unioned in."""
        import hermes_cli.models as hm

        live = ["glm-5.3", "glm-5.3-flash", "brand-new-model"]
        profile = _Profile(list(live))
        profile.fetch_models = lambda api_key=None, base_url=None, timeout=8.0: list(live)
        with mock.patch("providers.get_provider_profile", return_value=profile), \
             mock.patch("hermes_cli.auth.resolve_api_key_provider_credentials",
                        create=True,
                        return_value={"api_key": "k", "base_url": "https://x"}), \
             mock.patch.object(
                 hm, "_PROVIDER_MODELS",
                 {**hm._PROVIDER_MODELS,
                  "zai": ["glm-5.3", "DEAD-CURATED-MODEL"]},
             ), \
             mock.patch.dict(os.environ, {"GLM_API_KEY": "test-key-zai"}):
            got = hm.provider_model_ids("zai", force_refresh=True)
        self.assertEqual(got, live)
        self.assertNotIn("DEAD-CURATED-MODEL", got)

    def test_live_authoritative_set_contains_required_five(self):
        import hermes_cli.models as hm

        for slug in REQUIRED:
            self.assertIn(slug, hm._LIVE_AUTHORITATIVE_PICKER_PROVIDERS)


class ConfigNonRegression(unittest.TestCase):
    """Config YAML untouched by this work; aliases/fallbacks resolve."""

    def test_config_files_not_modified_by_this_task(self):
        base = json.load(open(os.path.join(
            _REAL_HERMES_HOME, "cache", "config_inventory",
            "baseline_hashes.json")))
        import hashlib

        checked = 0
        for rel, meta in base.items():
            if "error" in meta:
                continue
            path = os.path.join(_REAL_HERMES_HOME, rel)
            h = hashlib.sha256(open(path, "rb").read()).hexdigest()
            self.assertEqual(
                h, meta["sha256"],
                f"{rel} changed vs baseline (must remain untouched)")
            checked += 1
        self.assertGreater(checked, 20)

    def test_deepseek_v4_flash_alias_resolves_to_canonical(self):
        import yaml

        cfg = yaml.safe_load(open(os.path.join(_REAL_HERMES_HOME, "config.yaml")))
        alias = cfg["model_aliases"]["deepseek-v4-flash"]
        self.assertEqual(alias["model"], "deepseek-flash")
        self.assertEqual(alias["provider"], "deepseek")

    def test_default_model_and_fallback_order_unchanged(self):
        import yaml

        cfg = yaml.safe_load(open(os.path.join(_REAL_HERMES_HOME, "config.yaml")))
        self.assertEqual(cfg["model"]["provider"], "zai")
        self.assertEqual(cfg["providers"]["zai"]["default_model"], "glm-5.3")
        chain = [(f["provider"], f["model"]) for f in cfg["fallback_providers"]]
        self.assertEqual(
            chain,
            [("kilocode", "kilo-auto/free"),
             ("openrouter", "openrouter/free"),
             ("deepseek", "deepseek-flash")],
        )


if __name__ == "__main__":
    unittest.main()

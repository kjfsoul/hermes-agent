"""Tests for live+curated merge in the generic profile-based provider path.

Guards two contracts:

* #46850 — when a provider's live /v1/models endpoint returns a stale or
  incomplete list, the static curated models from ``_PROVIDER_MODELS`` must
  still appear in the merged result (nothing is dropped).
* #46309 / #49129 — merge *order* is per-provider. Single providers
  (kimi, zai) stay **curated-first** so a deliberately surfaced newest model
  leads even when the live API lags. ``_LIVE_FIRST_PICKER_PROVIDERS``
  (OpenCode Zen / Go) flip to **live-first** because their live API is the
  authoritative catalog and stale curated entries must not lead the picker.
"""

from unittest.mock import MagicMock, patch

from hermes_cli.models import (
    _LIVE_FIRST_PICKER_PROVIDERS,
    provider_model_ids,
)


class TestGenericProviderLiveCuratedMerge:
    """provider_model_ids merges live + curated for generic api_key providers."""

    def _make_profile(self, models=None):
        """Create a minimal mock provider profile."""
        p = MagicMock()
        p.auth_type = "api_key"
        p.base_url = "https://api.example.com/v1"
        p.fetch_models.return_value = models
        p.fallback_models = None
        return p

    def test_curated_first_for_single_provider(self):
        """Generic single providers stay curated-first; live-only appended.

        (kimi/zai slugs normalize to dedicated catalogs upstream; use a
        neutral provider slug that exercises the generic profile path.)
        """
        curated = ["ex-k2.5", "ex-k2.6", "ex-k3"]  # authoritative-intent order
        # Live API lags AND surfaces a brand-new model not yet curated.
        live = ["ex-k3", "ex-k4-preview"]
        profile = self._make_profile(live)

        with (
            patch("providers.get_provider_profile", return_value=profile),
            patch(
                "hermes_cli.auth.resolve_api_key_provider_credentials",
                return_value={"api_key": "k", "base_url": ""},
            ),
            patch.dict("hermes_cli.models._PROVIDER_MODELS", {"exampleai": curated}),
        ):
            result = provider_model_ids("exampleai")

        # Curated entries lead (commit 658ac1d86, #46309).
        assert result[: len(curated)] == curated
        # Live-only entries (kimi-k4-preview) still surface, appended afterwards.
        assert "ex-k4-preview" in result
        assert result.index("ex-k4-preview") >= len(curated)
        # No duplicates for models present in both.
        assert result.count("ex-k3") == 1

    def test_zai_is_live_authoritative(self):
        """model-refresh policy: zai's successful live catalog IS the picker.

        Curated-only entries (dead IDs) must NOT be unioned back in when the
        live fetch succeeded — a stale curated ID can never resurface after a
        successful refresh.
        """
        from hermes_cli.models import _LIVE_AUTHORITATIVE_PICKER_PROVIDERS

        assert "zai" in _LIVE_AUTHORITATIVE_PICKER_PROVIDERS
        curated = ["glm-5.2", "glm-5.1", "glm-5"]
        live = ["glm-5"]  # live dropped 5.2/5.1 (e.g. retired upstream)
        profile = self._make_profile(live)

        with (
            patch("providers.get_provider_profile", return_value=profile),
            patch(
                "hermes_cli.auth.resolve_api_key_provider_credentials",
                return_value={"api_key": "k", "base_url": ""},
            ),
            patch.dict("hermes_cli.models._PROVIDER_MODELS", {"zai": curated}),
        ):
            result = provider_model_ids("zai")

        assert result == live  # no curated-only resurrection


    def test_no_models_dropped_either_direction(self):
        """Every live AND curated model survives the merge for both modes."""
        live = ["a", "b"]
        # generic curated-first provider (zai is live-authoritative — see
        # test_zai_is_live_authoritative)
        with (
            patch("providers.get_provider_profile", return_value=self._make_profile(live)),
            patch(
                "hermes_cli.auth.resolve_api_key_provider_credentials",
                return_value={"api_key": "k", "base_url": ""},
            ),
            patch.dict("hermes_cli.models._PROVIDER_MODELS", {"exampleai": ["c", "b"]}),
        ):
            ex_result = set(provider_model_ids("exampleai"))
        assert {"a", "b", "c"} <= ex_result

        # opencode-zen = live-first
        with (
            patch("providers.get_provider_profile", return_value=self._make_profile(live)),
            patch(
                "hermes_cli.auth.resolve_api_key_provider_credentials",
                return_value={"api_key": "k", "base_url": ""},
            ),
            patch.dict("hermes_cli.models._PROVIDER_MODELS", {"opencode-zen": ["c", "b"]}),
        ):
            zen_result = set(provider_model_ids("opencode-zen"))
        assert {"a", "b", "c"} <= zen_result

    def test_opencode_go_merge_does_not_resurrect_delisted_model(self):
        """#95914 bug class, end-to-end through provider_model_ids with the REAL curated floor:
        the Go relay (GET /zen/go/v1/models) delisted ox-alpha-free 2026-09-09. The live-first
        merge must not resurrect it from the curated floor, or the picker keeps offering a model
        that now 401s (REVERT-PROOF: a stale floor re-adds it and this fails)."""
        assert "opencode-go" in _LIVE_FIRST_PICKER_PROVIDERS
        live = ["deepseek-v4-flash", "kimi-k3", "omen-alpha"]  # current Go relay (no ox-alpha-free)

        with (
            patch("providers.get_provider_profile", return_value=self._make_profile(live)),
            patch(
                "hermes_cli.auth.resolve_api_key_provider_credentials",
                return_value={"api_key": "k", "base_url": ""},
            ),
        ):
            result = provider_model_ids("opencode-go")

        assert "ox-alpha-free" not in result
        assert {"deepseek-v4-flash", "kimi-k3", "omen-alpha"} <= set(result)


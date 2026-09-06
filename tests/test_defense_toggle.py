"""The on/off controls for the defenses.

Three levels, narrowest first: an explicit value on the request, the
deployment default for requests that omit the field, and a master switch that
forces everything off. The master switch exists so a whole experiment can be
returned to the undefended baseline without touching any caller.
"""

import importlib

import services.orchestrator.app as orchestrator_module


def _reload(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return importlib.reload(orchestrator_module)


def test_omitted_defense_takes_the_deployment_default(monkeypatch) -> None:
    module = _reload(
        monkeypatch,
        DEFAULT_RETRIEVAL_DEFENSE="cluster",
        DEFAULT_GENERATION_DEFENSE="isolate_conflict",
    )
    assert module.resolve_retrieval_defense(None) == "cluster"
    assert module.resolve_generation_defense(None) == "isolate_conflict"


def test_explicit_none_turns_the_defense_off_for_one_request(monkeypatch) -> None:
    module = _reload(monkeypatch, DEFAULT_RETRIEVAL_DEFENSE="cluster")
    assert module.resolve_retrieval_defense("none") == "none"


def test_explicit_value_overrides_the_deployment_default(monkeypatch) -> None:
    module = _reload(monkeypatch, DEFAULT_RETRIEVAL_DEFENSE="cluster")
    assert module.resolve_retrieval_defense("ragmask") == "ragmask"


def test_master_switch_overrides_everything(monkeypatch) -> None:
    module = _reload(
        monkeypatch,
        DEFENSES_ENABLED="false",
        DEFAULT_RETRIEVAL_DEFENSE="cluster",
        DEFAULT_GENERATION_DEFENSE="robustrag",
    )
    assert module.resolve_retrieval_defense("ragpart") == "none"
    assert module.resolve_generation_defense("robustrag") == "none"
    assert module.resolve_retrieval_defense(None) == "none"


def test_defaults_are_off(monkeypatch) -> None:
    for key in (
        "DEFENSES_ENABLED",
        "DEFAULT_RETRIEVAL_DEFENSE",
        "DEFAULT_GENERATION_DEFENSE",
    ):
        monkeypatch.delenv(key, raising=False)
    module = importlib.reload(orchestrator_module)
    assert module.DEFENSES_ENABLED is True
    assert module.resolve_retrieval_defense(None) == "none"
    assert module.resolve_generation_defense(None) == "none"

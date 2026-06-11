"""
Test – Constraint Registry
============================
Test per il catalogo dei vincoli (constraint_registry.py).

Verifica che:
  1. Tutti i Known Cases abbiano i campi obbligatori
  2. Tutti gli Extension Pattern abbiano i campi obbligatori
  3. build_classifier_context() produca un testo coerente
  4. I tipi di estensione nel registro corrispondano ai tipi in ExtensionSpec
"""

from __future__ import annotations
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constraint_registry import (
    KNOWN_CONSTRAINTS,
    EXTENSION_PATTERNS,
    get_known_constraint_names,
    get_extension_pattern_names,
    build_classifier_context,
)
from models import ExtensionSpec


class TestKnownConstraints:

    def test_all_known_cases_have_required_fields(self):
        """Ogni Known Case deve avere i campi obbligatori."""
        required = {"label", "description", "parameters", "solver_handler", "unit_tested"}
        for name, kc in KNOWN_CONSTRAINTS.items():
            missing = required - set(kc.keys())
            assert not missing, f"Known Case '{name}' manca dei campi: {missing}"

    def test_known_cases_include_all_hard_constraints(self):
        """I vincoli H1-H7 devono essere tutti presenti nel registro."""
        names = get_known_constraint_names()
        # Verifica che i vincoli critici siano presenti
        assert "min_staff_per_shift"        in names
        assert "max_one_shift_per_day"      in names
        assert "no_night_to_morning_consecutive" in names
        assert "night_rest_days"            in names
        assert "workload_target"            in names
        assert "weekly_hours_cap"           in names
        assert "worker_unavailability"      in names

    def test_known_cases_include_soft_constraints(self):
        """I vincoli soft S1-S5 devono essere nel registro."""
        names = get_known_constraint_names()
        assert "preferred_shift_bonus"  in names
        assert "avoided_shift_penalty"  in names
        assert "night_tolerance"        in names
        assert "holiday_tolerance"      in names
        assert "preferred_rest_day"     in names

    def test_all_known_cases_are_unit_tested(self):
        """Tutti i Known Cases devono dichiarare unit_tested=True."""
        for name, kc in KNOWN_CONSTRAINTS.items():
            assert kc.get("unit_tested") is True, (
                f"Known Case '{name}' dichiara unit_tested=False o non lo dichiara."
            )


class TestExtensionPatterns:

    def test_all_patterns_have_required_fields(self):
        """Ogni Extension Pattern deve avere i campi obbligatori."""
        required = {"label", "description", "parameters", "solver_function", "unit_tested", "risk_level"}
        for name, ep in EXTENSION_PATTERNS.items():
            missing = required - set(ep.keys())
            assert not missing, f"Extension Pattern '{name}' manca dei campi: {missing}"

    def test_all_patterns_are_unit_tested(self):
        """Tutti gli Extension Pattern devono dichiarare unit_tested=True."""
        for name, ep in EXTENSION_PATTERNS.items():
            assert ep.get("unit_tested") is True, (
                f"Extension Pattern '{name}' dichiara unit_tested=False."
            )

    def test_risk_levels_are_valid(self):
        """I risk_level devono essere uno tra LOW, MEDIUM, HIGH."""
        valid_levels = {"LOW", "MEDIUM", "HIGH"}
        for name, ep in EXTENSION_PATTERNS.items():
            assert ep.get("risk_level") in valid_levels, (
                f"Extension Pattern '{name}' ha risk_level non valido: {ep.get('risk_level')}"
            )

    def test_extension_pattern_keys_match_pydantic_literal(self):
        """
        I tipi nel registro devono corrispondere esattamente ai Literal valori
        in ExtensionSpec.extension_type. Questo test previene disallineamenti
        tra il registro e il modello Pydantic.
        """
        # Estrai i valori ammessi da ExtensionSpec tramite il tipo Pydantic
        import typing
        hints = typing.get_type_hints(ExtensionSpec)
        ext_type_hint = hints.get("extension_type", None)

        if ext_type_hint is None:
            pytest.skip("Impossibile accedere al type hint di ExtensionSpec.extension_type")

        # Ottieni i valori del Literal
        literal_args = set()
        if hasattr(ext_type_hint, "__args__"):
            literal_args = set(ext_type_hint.__args__)
        else:
            # Python 3.8+ annotation evaluation
            try:
                literal_args = set(typing.get_args(ext_type_hint))
            except Exception:
                pytest.skip("Impossibile estrarre i Literal args da ExtensionSpec")

        registry_keys = set(EXTENSION_PATTERNS.keys())

        # Ogni pattern nel registro deve avere un Literal corrispondente
        unregistered = registry_keys - literal_args
        assert not unregistered, (
            f"Extension Pattern nel registro ma non in ExtensionSpec.extension_type: {unregistered}. "
            "Aggiornare models.py."
        )

        # Ogni Literal in ExtensionSpec deve avere un pattern nel registro
        undocumented = literal_args - registry_keys
        assert not undocumented, (
            f"Tipi in ExtensionSpec.extension_type ma non nel registro: {undocumented}. "
            "Aggiornare constraint_registry.py."
        )


class TestBuildClassifierContext:

    def test_context_is_non_empty_string(self):
        ctx = build_classifier_context()
        assert isinstance(ctx, str)
        assert len(ctx) > 100

    def test_context_contains_known_cases(self):
        ctx = build_classifier_context()
        assert "VINCOLI NOTI" in ctx
        assert "min_staff_per_shift" in ctx

    def test_context_contains_extension_patterns(self):
        ctx = build_classifier_context()
        assert "EXTENSION PATTERN" in ctx
        assert "max_consecutive_nights" in ctx

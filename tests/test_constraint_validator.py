"""
Test – Constraint Validator
==============================
Test per il validatore strutturale (constraint_validator.py).

Verifica che:
  1. Parametri validi passino la validazione
  2. Parametri fuori range generino errori
  3. Worker ID inesistenti vengano rilevati
  4. Il report aggregato rifletta tutti gli errori
  5. Il check di compatibilità rilevi cicli e combinazioni rischiose
"""

from __future__ import annotations
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ExtensionSpec, ShiftPreference
from constraint_validator import (
    validate_extension,
    validate_all_extensions,
    check_extension_compatibility,
    ValidationResult,
    ValidationReport,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_worker(worker_id: str, worker_type: str = "standard") -> ShiftPreference:
    return ShiftPreference(
        worker_id=worker_id,
        worker_name=f"Worker {worker_id}",
        worker_type=worker_type,
        preferred_shifts=[],
        avoided_shifts=[],
        night_tolerance=True,
        holiday_tolerance=True,
        unavailable_days_of_week=[],
        preferred_rest_day=None,
        emergency_coverage=0,
    )


@pytest.fixture
def workers():
    return [_make_worker(f"W{i:02d}") for i in range(1, 6)]


# ── Test validazione max_consecutive_nights ────────────────────────────────────

class TestValidateMaxConsecutiveNights:

    def test_valid_params(self, workers):
        ext = ExtensionSpec(
            extension_type="max_consecutive_nights",
            parameters={"max_consecutive": 3},
        )
        result = validate_extension(ext, workers)
        assert result.passed, f"Atteso passed=True, errori: {result.errors}"

    def test_missing_param(self, workers):
        ext = ExtensionSpec(
            extension_type="max_consecutive_nights",
            parameters={},
        )
        result = validate_extension(ext, workers)
        assert not result.passed
        assert any("max_consecutive" in e for e in result.errors)

    def test_out_of_range_low(self, workers):
        ext = ExtensionSpec(
            extension_type="max_consecutive_nights",
            parameters={"max_consecutive": 0},
        )
        result = validate_extension(ext, workers)
        assert not result.passed

    def test_out_of_range_high(self, workers):
        ext = ExtensionSpec(
            extension_type="max_consecutive_nights",
            parameters={"max_consecutive": 8},
        )
        result = validate_extension(ext, workers)
        assert not result.passed

    def test_warning_for_max1(self, workers):
        ext = ExtensionSpec(
            extension_type="max_consecutive_nights",
            parameters={"max_consecutive": 1},
        )
        result = validate_extension(ext, workers)
        assert result.passed  # è valido ma genera un avviso
        assert len(result.warnings) > 0


# ── Test validazione pair_always_same_shift ────────────────────────────────────

class TestValidatePairAlwaysSameShift:

    def test_valid_pair(self, workers):
        ext = ExtensionSpec(
            extension_type="pair_always_same_shift",
            parameters={"worker_a_id": "W01", "worker_b_id": "W02"},
        )
        result = validate_extension(ext, workers)
        assert result.passed

    def test_unknown_worker_a(self, workers):
        ext = ExtensionSpec(
            extension_type="pair_always_same_shift",
            parameters={"worker_a_id": "XXXXXXXXXXX", "worker_b_id": "W02"},
        )
        result = validate_extension(ext, workers)
        assert not result.passed
        assert any("worker_a_id" in e for e in result.errors)

    def test_same_worker_error(self, workers):
        ext = ExtensionSpec(
            extension_type="pair_always_same_shift",
            parameters={"worker_a_id": "W01", "worker_b_id": "W01"},
        )
        result = validate_extension(ext, workers)
        assert not result.passed

    def test_missing_params(self, workers):
        ext = ExtensionSpec(
            extension_type="pair_always_same_shift",
            parameters={},
        )
        result = validate_extension(ext, workers)
        assert not result.passed
        assert len(result.errors) >= 2  # entrambi i parametri mancanti


# ── Test validazione max_shifts_of_type_per_week ───────────────────────────────

class TestValidateMaxShiftsOfTypePerWeek:

    def test_valid_params(self, workers):
        ext = ExtensionSpec(
            extension_type="max_shifts_of_type_per_week",
            parameters={"shift_type": "night", "max_per_week": 2},
        )
        result = validate_extension(ext, workers)
        assert result.passed

    def test_invalid_shift_type(self, workers):
        ext = ExtensionSpec(
            extension_type="max_shifts_of_type_per_week",
            parameters={"shift_type": "evening", "max_per_week": 2},
        )
        result = validate_extension(ext, workers)
        assert not result.passed

    def test_max_per_week_out_of_range(self, workers):
        ext = ExtensionSpec(
            extension_type="max_shifts_of_type_per_week",
            parameters={"shift_type": "morning", "max_per_week": 10},
        )
        result = validate_extension(ext, workers)
        assert not result.passed

    def test_applies_to_unknown_worker(self, workers):
        ext = ExtensionSpec(
            extension_type="max_shifts_of_type_per_week",
            parameters={
                "shift_type": "morning",
                "max_per_week": 3,
                "applies_to": ["W01", "XXXXXX"],
            },
        )
        result = validate_extension(ext, workers)
        assert not result.passed


# ── Test ValidationReport aggregato ──────────────────────────────────────────

class TestValidateAllExtensions:

    def test_all_valid(self, workers):
        extensions = [
            ExtensionSpec(
                extension_type="max_consecutive_nights",
                parameters={"max_consecutive": 3},
            ),
            ExtensionSpec(
                extension_type="max_shifts_of_type_per_week",
                parameters={"shift_type": "night", "max_per_week": 2},
            ),
        ]
        report = validate_all_extensions(extensions, workers)
        assert report.all_passed
        assert len(report.results) == 2

    def test_one_invalid_fails_all(self, workers):
        extensions = [
            ExtensionSpec(
                extension_type="max_consecutive_nights",
                parameters={"max_consecutive": 3},
            ),
            ExtensionSpec(
                extension_type="pair_always_same_shift",
                parameters={"worker_a_id": "ZZZZZ", "worker_b_id": "W02"},
            ),
        ]
        report = validate_all_extensions(extensions, workers)
        assert not report.all_passed
        assert len(report.errors) > 0

    def test_empty_extensions_passes(self, workers):
        report = validate_all_extensions([], workers)
        assert report.all_passed


# ── Test compatibilità tra estensioni ─────────────────────────────────────────

class TestExtensionCompatibility:

    def test_duplicate_worker_in_pairs_generates_warning(self):
        """W01 in due coppie diverse → avviso di potenziale ciclo."""
        extensions = [
            ExtensionSpec(
                extension_type="pair_always_same_shift",
                parameters={"worker_a_id": "W01", "worker_b_id": "W02"},
            ),
            ExtensionSpec(
                extension_type="pair_always_same_shift",
                parameters={"worker_a_id": "W01", "worker_b_id": "W03"},
            ),
        ]
        warnings = check_extension_compatibility(extensions)
        assert len(warnings) > 0
        assert any("W01" in w for w in warnings)

    def test_no_compatibility_issues(self):
        """Estensioni non conflittuali non devono generare avvisi."""
        extensions = [
            ExtensionSpec(
                extension_type="max_consecutive_nights",
                parameters={"max_consecutive": 3},
            ),
            ExtensionSpec(
                extension_type="max_shifts_of_type_per_week",
                parameters={"shift_type": "morning", "max_per_week": 4},
            ),
        ]
        warnings = check_extension_compatibility(extensions)
        # Non dovrebbero esserci avvisi per questa combinazione
        assert not any("ciclo" in w.lower() for w in warnings)

    def test_max_consecutive_1_generates_feasibility_warning(self):
        """max_consecutive=1 deve generare un avviso di fattibilità."""
        extensions = [
            ExtensionSpec(
                extension_type="max_consecutive_nights",
                parameters={"max_consecutive": 1},
            ),
        ]
        warnings = check_extension_compatibility(extensions)
        assert any("max_consecutive_nights=1" in w for w in warnings)

"""
Unit tests for the Financial Stress Index formula.
Run locally with: pytest tests/test_financial_stress_index.py -v
No Spark required — pure Python.
"""

import pytest


# ── Standalone FSI implementation (mirrors 02_transform_silver.py) ───────────
UNEMP_MIN = 7.0
UNEMP_MAX = 16.2


def compute_fsi(
    debtor: int,
    tuition_fees_up_to_date: int,
    scholarship_holder: int,
    displaced: int,
    unemployment_rate: float,
    units_enrolled_1st: int,
    units_without_eval_1st: int,
) -> float:
    """
    Financial Stress Index in [0, 1].

    Weights:
      - debt_burden         (debtor == 1)              × 0.30
      - payment_failure     (tuition_fees == 0)        × 0.25
      - no_scholarship      (scholarship_holder == 0)  × 0.15
      - displacement        (displaced == 1)            × 0.10
      - macro_unemployment  (normalised rate)           × 0.10
      - academic_disengage  (missed eval ratio)         × 0.10
    """
    # Normalise unemployment to [0, 1]
    unemp_norm = (unemployment_rate - UNEMP_MIN) / (UNEMP_MAX - UNEMP_MIN)
    unemp_norm = max(0.0, min(1.0, unemp_norm))

    # Academic disengagement
    if units_enrolled_1st == 0:
        disengagement = 0.0
    else:
        disengagement = units_without_eval_1st / units_enrolled_1st
        disengagement = max(0.0, min(1.0, disengagement))

    fsi = (
        debtor                           * 0.30 +
        (1 - tuition_fees_up_to_date)    * 0.25 +
        (1 - scholarship_holder)         * 0.15 +
        displaced                        * 0.10 +
        unemp_norm                       * 0.10 +
        disengagement                    * 0.10
    )
    return round(fsi, 6)


# ── Tests ────────────────────────────────────────────────────────────────────

class TestFSIBoundaries:
    def test_zero_stress(self):
        """Student with zero stress on all dimensions should return 0."""
        fsi = compute_fsi(
            debtor=0,
            tuition_fees_up_to_date=1,
            scholarship_holder=1,
            displaced=0,
            unemployment_rate=UNEMP_MIN,  # minimum unemployment → unemp_norm = 0
            units_enrolled_1st=5,
            units_without_eval_1st=0,
        )
        assert fsi == pytest.approx(0.0, abs=1e-5), f"Expected 0.0 but got {fsi}"

    def test_max_stress(self):
        """Student with maximum stress on all dimensions should return 1."""
        fsi = compute_fsi(
            debtor=1,
            tuition_fees_up_to_date=0,
            scholarship_holder=0,
            displaced=1,
            unemployment_rate=UNEMP_MAX,  # maximum unemployment → unemp_norm = 1
            units_enrolled_1st=5,
            units_without_eval_1st=5,     # 100% disengagement
        )
        assert fsi == pytest.approx(1.0, abs=1e-5), f"Expected 1.0 but got {fsi}"

    def test_fsi_always_in_range(self):
        """FSI must always be in [0, 1]."""
        test_cases = [
            (0, 1, 1, 0, 7.0,  5, 0),
            (1, 0, 0, 1, 16.2, 5, 5),
            (1, 1, 0, 0, 10.0, 3, 1),
            (0, 0, 1, 1, 8.5,  0, 0),
        ]
        for args in test_cases:
            fsi = compute_fsi(*args)
            assert 0.0 <= fsi <= 1.0, f"FSI out of range for args {args}: {fsi}"


class TestFSIWeights:
    def test_debt_burden_weight(self):
        """Debtor=1 vs Debtor=0 should differ by 0.30 (all else equal)."""
        base = dict(
            tuition_fees_up_to_date=1, scholarship_holder=1,
            displaced=0, unemployment_rate=UNEMP_MIN,
            units_enrolled_1st=5, units_without_eval_1st=0,
        )
        fsi_no_debt = compute_fsi(debtor=0, **base)
        fsi_debt    = compute_fsi(debtor=1, **base)
        assert fsi_debt - fsi_no_debt == pytest.approx(0.30, abs=1e-5)

    def test_tuition_overdue_weight(self):
        """Tuition overdue (=0) vs up to date (=1) should differ by 0.25."""
        base = dict(
            debtor=0, scholarship_holder=1,
            displaced=0, unemployment_rate=UNEMP_MIN,
            units_enrolled_1st=5, units_without_eval_1st=0,
        )
        fsi_paid    = compute_fsi(tuition_fees_up_to_date=1, **base)
        fsi_overdue = compute_fsi(tuition_fees_up_to_date=0, **base)
        assert fsi_overdue - fsi_paid == pytest.approx(0.25, abs=1e-5)

    def test_no_scholarship_weight(self):
        """No scholarship vs scholarship should differ by 0.15."""
        base = dict(
            debtor=0, tuition_fees_up_to_date=1,
            displaced=0, unemployment_rate=UNEMP_MIN,
            units_enrolled_1st=5, units_without_eval_1st=0,
        )
        fsi_scholar  = compute_fsi(scholarship_holder=1, **base)
        fsi_no_schol = compute_fsi(scholarship_holder=0, **base)
        assert fsi_no_schol - fsi_scholar == pytest.approx(0.15, abs=1e-5)


class TestFSIEdgeCases:
    def test_zero_enrolled_units_no_division_error(self):
        """Zero enrolled units should not cause division by zero."""
        fsi = compute_fsi(
            debtor=0, tuition_fees_up_to_date=1, scholarship_holder=1,
            displaced=0, unemployment_rate=UNEMP_MIN,
            units_enrolled_1st=0,
            units_without_eval_1st=0,
        )
        assert fsi == pytest.approx(0.0, abs=1e-5)

    def test_unemployment_below_min_clipped(self):
        """Unemployment below historical minimum should be clipped to 0."""
        fsi = compute_fsi(
            debtor=0, tuition_fees_up_to_date=1, scholarship_holder=1,
            displaced=0, unemployment_rate=1.0,   # far below UNEMP_MIN
            units_enrolled_1st=5, units_without_eval_1st=0,
        )
        assert fsi == pytest.approx(0.0, abs=1e-5)

    def test_unemployment_above_max_clipped(self):
        """Unemployment above historical maximum should contribute at most 0.10."""
        base = dict(
            debtor=0, tuition_fees_up_to_date=1, scholarship_holder=1,
            displaced=0, units_enrolled_1st=5, units_without_eval_1st=0,
        )
        fsi_at_max  = compute_fsi(unemployment_rate=UNEMP_MAX,   **base)
        fsi_over_max = compute_fsi(unemployment_rate=UNEMP_MAX + 10, **base)
        assert fsi_at_max == pytest.approx(fsi_over_max, abs=1e-5)

    def test_high_risk_student_profile(self):
        """
        Typical high-risk student: debtor, overdue tuition, no scholarship,
        displaced, high unemployment, high disengagement → FSI >= 0.6
        """
        fsi = compute_fsi(
            debtor=1,
            tuition_fees_up_to_date=0,
            scholarship_holder=0,
            displaced=1,
            unemployment_rate=14.0,
            units_enrolled_1st=6,
            units_without_eval_1st=4,
        )
        assert fsi >= 0.6, f"Expected high FSI >= 0.6, got {fsi}"

    def test_low_risk_student_profile(self):
        """
        Low-risk student: no debt, pays tuition, has scholarship,
        not displaced, low unemployment, fully engaged → FSI < 0.3
        """
        fsi = compute_fsi(
            debtor=0,
            tuition_fees_up_to_date=1,
            scholarship_holder=1,
            displaced=0,
            unemployment_rate=8.0,
            units_enrolled_1st=6,
            units_without_eval_1st=0,
        )
        assert fsi < 0.3, f"Expected low FSI < 0.3, got {fsi}"

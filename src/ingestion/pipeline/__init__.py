"""Deterministic extraction pipeline: PDF evidence → simulation plans.

Pipeline stages:
  extract_evidence → build_experiment_candidates → extract_scenarios
  → enrich_scenarios → validate_scenarios → validate_plan
  → build_plans_from_scenarios
"""

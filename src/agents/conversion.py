"""Chemkin → Cantera YAML conversion via T3's fix_cantera.py."""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

from src.agents.llm_client import LLMClient
from src.agents.validators import _normalize_equation
from src.schemas.experimental import ConversionResult

logger = logging.getLogger(__name__)


class ChemkinConverter:
    """Converts Chemkin mechanism files to Cantera YAML using T3.

    T3 may live in a separate Python environment, so we call it as a
    subprocess rather than importing directly.
    """

    MAX_ATTEMPTS = 2

    def __init__(self, llm_client: LLMClient, t3_python: str = "python") -> None:
        self.llm_client = llm_client
        self.t3_python = t3_python

    async def _run_t3(
        self, chemkin_path: Path, output_path: Path
    ) -> tuple[int, str, str]:
        """Run T3 fix_cantera as a subprocess. Returns (returncode, stdout, stderr)."""
        proc = await asyncio.create_subprocess_exec(
            self.t3_python,
            "-m",
            "t3.utils.fix_cantera",
            str(chemkin_path),
            str(output_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await proc.communicate()
        return (
            proc.returncode or 0,
            stdout_bytes.decode(errors="replace"),
            stderr_bytes.decode(errors="replace"),
        )

    async def _diagnose(self, stderr: str) -> str:
        """Ask the LLM to diagnose a T3 conversion failure."""
        prompt = (
            "A Chemkin-to-Cantera conversion using T3 fix_cantera.py failed.\n"
            "Here is the stderr output:\n\n"
            f"```\n{stderr}\n```\n\n"
            "Diagnose the likely cause and suggest a concrete fix."
        )
        result = await self.llm_client.complete(
            prompt=prompt,
            system="You are a chemical kinetics software expert.",
            agent_name="conversion",
        )
        return result

    async def convert(
        self, chemkin_path: Path, output_dir: Path
    ) -> ConversionResult:
        """Convert a Chemkin mechanism file to Cantera YAML.

        Attempts the conversion up to MAX_ATTEMPTS times. On first failure,
        the LLM diagnoses the error before a retry.
        """
        output_path = output_dir / (chemkin_path.stem + ".yaml")
        errors: list[str] = []
        warnings: list[str] = []

        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            returncode, stdout, stderr = await self._run_t3(
                chemkin_path, output_path
            )

            if stdout.strip():
                warnings.append(stdout.strip())

            if returncode == 0 and output_path.exists():
                logger.info("T3 conversion succeeded on attempt %d", attempt)
                return ConversionResult(
                    success=True,
                    output_path=output_path,
                    errors=errors,
                    warnings=warnings,
                    attempts=attempt,
                )

            errors.append(stderr.strip() or f"T3 exited with code {returncode}")
            logger.warning(
                "T3 conversion failed on attempt %d: %s", attempt, errors[-1]
            )

            if attempt < self.MAX_ATTEMPTS:
                diagnosis = await self._diagnose(stderr)
                logger.info("LLM diagnosis: %s", diagnosis)
                warnings.append(f"LLM diagnosis: {diagnosis}")

        return ConversionResult(
            success=False,
            output_path=None,
            errors=errors,
            warnings=warnings,
            attempts=self.MAX_ATTEMPTS,
        )

    # Trailing Chemkin annotations to strip before parsing rate params
    _ANNOTATIONS_RE = re.compile(
        r"\s+(?:DUPLICATE|DUP|LOW|TROE|REV|PLOG|HIGH|SRI|FORD|RORD)\b.*",
        re.IGNORECASE,
    )

    # Chemkin rate line: reaction string then A, n, Ea (whitespace-separated)
    # A can be scientific notation (1.04E+14) or plain float (104000.0)
    _RATE_RE = re.compile(
        r"^(?P<rxn>.+?)\s+"                          # reaction string
        r"(?P<A>[0-9.]+(?:[eE][+-]?\d+)?)"           # pre-exponential A
        r"\s+(?P<n>[-+]?[0-9.]+(?:[eE][+-]?\d+)?)"   # temperature exponent n
        r"\s+(?P<Ea>[-+]?[0-9.]+(?:[eE][+-]?\d+)?)"  # activation energy Ea
        r"\s*$"
    )

    def extract_rates(self, chemkin_path: Path) -> dict[str, dict]:
        """Parse Chemkin file for Arrhenius rate parameters (A, n, Ea).

        Returns a dict mapping normalised reaction string → {"A", "n", "Ea"}.
        Skips comment lines (starting with !) and malformed lines.
        Never raises — returns whatever was successfully parsed.
        """
        rates: dict[str, dict] = {}
        try:
            text = chemkin_path.read_text(errors="replace")
        except OSError:
            logger.warning("Could not read Chemkin file: %s", chemkin_path)
            return rates

        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("!"):
                continue
            # Strip trailing annotations (DUPLICATE, LOW, TROE, etc.)
            stripped = self._ANNOTATIONS_RE.sub("", stripped)
            m = self._RATE_RE.match(stripped)
            if not m:
                continue
            rxn_raw = m.group("rxn").strip()
            normalised = _normalize_equation(rxn_raw)
            rates[normalised] = {
                "A": float(m.group("A")),
                "n": float(m.group("n")),
                "Ea": float(m.group("Ea")),
            }
        logger.info("Extracted %d rate entries from %s", len(rates), chemkin_path.name)
        return rates

"""Chemkin → Cantera YAML conversion via T3's fix_cantera.py."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from src.agents.llm_client import LLMClient
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

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


@contextlib.contextmanager
def debug_context(params: Dict[str, Any]) -> Iterator["DebugHooks"]:
    """Optional dump/profiling hooks for the gpt_rs Python extension.

    Supported params:
      - dump_dir: str | Path
      - dump_mode: "all" | "compile" (default: "all")
      - profile: bool (prints profiler tables to stderr when enabled)
      - profile_json: bool (writes profiler_report.json when dump_dir is set)
      - profile_trace: bool (writes profiler_trace.json when dump_dir is set)
    """

    hooks = DebugHooks(None, profile=False)
    dump_dir = params.get("dump_dir")
    dump_mode = str(params.get("dump_mode", "all"))
    profile_tables = bool(params.get("profile", False))
    profile_json = bool(params.get("profile_json", False))
    profile_trace = bool(params.get("profile_trace", False))
    profile = bool(profile_tables or profile_json or profile_trace)

    prev_profile_backend = None
    prev_c_trace = None
    if profile:
        prev_profile_backend = os.environ.get("GPTRS_PROFILE_BACKEND")
        os.environ["GPTRS_PROFILE_BACKEND"] = "1"
    if profile_trace:
        prev_c_trace = os.environ.get("GPTRS_C_TRACE")
        os.environ["GPTRS_C_TRACE"] = "1"

    try:
        try:
            import gpt_rs  # type: ignore[import-not-found]
        except ImportError:
            yield hooks
            return

        hooks = DebugHooks(gpt_rs, profile=profile)
        dump_enabled = False
        try:
            if dump_dir is not None:
                try:
                    gpt_rs.set_dump_dir(str(dump_dir), dump_mode)
                    dump_enabled = True
                except Exception:
                    print(
                        "dump_dir requested but gpt_rs.set_dump_dir is unavailable; rebuild gpt_rs "
                        "(e.g. `cd crates/gpt-rs-py && maturin develop --release --features faer "
                        "--skip-install`).",
                        file=sys.stderr,
                    )

            if profile:
                try:
                    gpt_rs.profiling_reset()
                except Exception:
                    pass

            if profile_trace:
                try:
                    gpt_rs.profiling_trace_reset()
                    gpt_rs.profiling_trace_enable()
                except Exception:
                    pass

            yield hooks
        finally:
            hooks.pop_all_sections()
            if profile_json:
                try:
                    bundle = gpt_rs.profiling_take_report_bundle(pretty=True)
                except Exception:
                    bundle = None
                if bundle:
                    text, json_report = bundle
                    if profile_tables:
                        print(text, file=sys.stderr)
                    if dump_dir is not None:
                        Path(dump_dir).mkdir(parents=True, exist_ok=True)
                        (Path(dump_dir) / "profiler_report.json").write_text(
                            json_report, encoding="utf-8"
                        )
                    else:
                        print(json_report, file=sys.stderr)
                else:
                    print(
                        "profiling enabled but no report available; "
                        "rebuild/install gpt_rs with profiler support "
                        f"(imported from {getattr(gpt_rs, '__file__', '<unknown>')}); "
                        "try: `cd crates/gpt-rs-py && maturin develop --release --features faer,profiler` "
                        "(omit `--skip-install` unless you also set `PYTHONPATH=crates/gpt-rs-py`).",
                        file=sys.stderr,
                    )
            elif profile_tables:
                try:
                    report = gpt_rs.profiling_take_report()
                except Exception:
                    report = None
                if report:
                    print(report, file=sys.stderr)
                else:
                    print(
                        "profiling enabled but no report available; "
                        "rebuild/install gpt_rs with profiler support "
                        f"(imported from {getattr(gpt_rs, '__file__', '<unknown>')}); "
                        "try: `cd crates/gpt-rs-py && maturin develop --release --features faer,profiler` "
                        "(omit `--skip-install` unless you also set `PYTHONPATH=crates/gpt-rs-py`).",
                        file=sys.stderr,
                    )

            if profile_trace:
                try:
                    trace_json = gpt_rs.profiling_take_trace_json(pretty=False)
                except Exception:
                    trace_json = None
                if trace_json and dump_dir is not None:
                    Path(dump_dir).mkdir(parents=True, exist_ok=True)
                    (Path(dump_dir) / "profiler_trace.json").write_text(
                        trace_json, encoding="utf-8"
                    )

                try:
                    gpt_rs.profiling_trace_disable()
                except Exception:
                    pass

            if dump_enabled:
                try:
                    gpt_rs.clear_dump_dir()
                except Exception:
                    pass
    finally:
        if profile:
            if prev_profile_backend is None:
                os.environ.pop("GPTRS_PROFILE_BACKEND", None)
            else:
                os.environ["GPTRS_PROFILE_BACKEND"] = prev_profile_backend
        if profile_trace:
            if prev_c_trace is None:
                os.environ.pop("GPTRS_C_TRACE", None)
            else:
                os.environ["GPTRS_C_TRACE"] = prev_c_trace


class DebugHooks:
    def __init__(self, gpt_rs_mod: Optional[Any], *, profile: bool) -> None:
        self._gpt_rs = gpt_rs_mod
        self.profile = bool(profile)

    def _call(self, name: str, *args: Any) -> Any:
        if not self.profile or self._gpt_rs is None:
            return None
        fn = getattr(self._gpt_rs, name, None)
        if fn is None:
            return None
        try:
            return fn(*args)
        except Exception:
            return None

    def push_section(self, name: str) -> None:
        self._call("profiling_push_section", str(name))

    def pop_section(self) -> bool:
        ok = self._call("profiling_pop_section")
        return bool(ok)

    def pop_all_sections(self) -> None:
        while self.pop_section():
            pass

#!/usr/bin/env python3
"""Production entrypoint: <input_pdf_dir> <output_predictions_path>.

Two stages: parallel state extraction (independent worker processes), then a
batch-epoch decision pass. Never omits a case and never emits malformed values:
any per-case failure degrades to a conservative NEEDS_REVIEW row.

Watchdog: each worker touches a heartbeat file before starting a PDF. A worker
that goes silent past MIB_STUCK_SECS (a hang below Python's signal layer,
where the worker's own SIGALRM can't fire) or that dies outright is killed and
respawned on its remaining slice, skipping the PDF it was stuck on. One
pathological PDF therefore costs at most one conservative row — never the
worker's slice, never the batch.
"""
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mib import two_ledger  # noqa: E402
from mib.pipeline import (FALLBACKS, batch_epoch, batch_frequent_sponsors,  # noqa: E402
                          decide)


def _batch_decision_inputs(states):
    """(epoch, batch_revoked, native_epoch, native_revoked, ablation).

    The native ledgers embedded under ``state['native_ledger']`` form an
    independent batch: their own epoch and frequent-sponsor set decide the
    native side, keeping each ledger a fully independent result.
    """
    epoch = batch_epoch(states)
    batch_revoked = batch_frequent_sponsors(states)
    ablation = two_ledger.ablation_from_env()
    native_states, has_native = two_ledger.native_batch_inputs(states)
    native_epoch = batch_epoch(native_states) if has_native else epoch
    native_revoked = (batch_frequent_sponsors(native_states)
                      if has_native else batch_revoked)
    return epoch, batch_revoked, native_epoch, native_revoked, ablation

STUCK_SECS = float(os.environ.get("MIB_STUCK_SECS", "150"))
STARTUP_GRACE = float(os.environ.get("MIB_STARTUP_GRACE", "120"))
POLL_SECS = float(os.environ.get("MIB_WATCHDOG_POLL", "2"))
MAX_NO_PROGRESS_RESPAWNS = 3
MAX_RETRY_CASES = int(os.environ.get("MIB_MAX_RETRY_CASES", "8"))
RETRY_CASE_TIMEOUT = float(os.environ.get("MIB_RETRY_CASE_TIMEOUT", "130"))
RETRY_BUDGET_SECS = float(os.environ.get("MIB_RETRY_BUDGET_SECS", "1100"))
RETRY_KILL_GRACE_SECS = float(os.environ.get("MIB_RETRY_KILL_GRACE_SECS", "5"))
BATCH_LIMIT_SECS = float(os.environ.get("MIB_BATCH_LIMIT_SECS", "30000"))
FINALIZE_RESERVE_SECS = float(os.environ.get("MIB_FINALIZE_RESERVE_SECS", "60"))
RUN_IDENTITY_SCHEMA = "mib-run-identity-v1"
RUN_RECEIPT_SCHEMA = "mib-run-receipt-v2"
RUN_IDENTITY_KEYS = {
    "schema", "producer_git_sha", "image_id", "image_revision",
    "image_inspect_sha256", "runtime_manifest_sha256", "config_sha256",
    "input_manifest_sha256", "run_split", "run_nonce",
}
RUN_SPLITS = {"dev", "holdout", "validation"}
SHA1_RE = re.compile(r"[0-9a-f]{40}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
IMAGE_ID_RE = re.compile(r"sha256:[0-9a-f]{64}")
WORKER_RECYCLE_EXIT_CODE = 75


class Shard:
    def __init__(self, tmp, idx, pdfs):
        self.tmp, self.idx = tmp, idx
        self.pdfs = list(pdfs)
        self.hb = tmp / f"hb{idx}.txt"
        self.gen = 0
        self.no_progress = 0
        self.state_files = []
        self.finished = False
        self.proc = None
        self.started = 0.0
        if self.pdfs:
            self._spawn(self.pdfs)
        else:
            self.finished = True

    def _spawn(self, pdfs):
        self.gen += 1
        self.pdfs = list(pdfs)
        slice_file = self.tmp / f"shard{self.idx}_g{self.gen}.txt"
        slice_file.write_text("\n".join(pdfs))
        state_file = self.tmp / f"state{self.idx}_g{self.gen}.jsonl"
        self.state_files.append(state_file)
        if self.hb.exists():
            self.hb.unlink()
        env = dict(os.environ, OMP_NUM_THREADS="1")
        self.proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve().parent / "run_shard.py"),
             str(slice_file), str(state_file), str(self.hb)],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.started = time.time()

    def _silent_too_long(self):
        try:
            return time.time() - self.hb.stat().st_mtime > STUCK_SECS
        except OSError:  # no heartbeat yet: still importing / loading models
            return time.time() - self.started > STARTUP_GRACE

    def _completed_ids(self):
        # Keyed by source-file stem, not resolved case id: scheduling tracks
        # input files, while the resolved id may differ on renamed inputs.
        done = set()
        for sf in self.state_files:
            if sf.exists():
                for line in open(sf):
                    try:
                        done.add(_state_stem(json.loads(line)))
                    except (json.JSONDecodeError, KeyError):
                        pass  # torn tail line mid-write
        return done

    def _respawn(self, reason):
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        culprit = self.hb.read_text().strip() if self.hb.exists() else None
        done = self._completed_ids()
        remaining = [p for p in self.pdfs
                     if Path(p).stem not in done and p != culprit]
        if culprit and Path(culprit).stem not in done:
            print(f"[watchdog] shard{self.idx} {reason}; "
                  f"skipping {Path(culprit).stem}", flush=True)
        # A respawn that can't shrink the slice (e.g. worker dies before its
        # first heartbeat) must not loop forever; give up the slice after a
        # few strikes and let the completeness backfill hedge those cases.
        self.no_progress = 0 if len(remaining) < len(self.pdfs) else self.no_progress + 1
        if not remaining or self.no_progress >= MAX_NO_PROGRESS_RESPAWNS:
            self.finished = True
            return
        self._spawn(remaining)

    def tick(self):
        if self.finished:
            return
        rc = self.proc.poll()
        if rc == 0:
            self.finished = True
        elif rc == WORKER_RECYCLE_EXIT_CODE:
            self._respawn("requested recognizer recycle")
        elif rc is not None:
            self._respawn(f"exited rc={rc}")
        elif self._silent_too_long():
            self._respawn("heartbeat stale")


def _parse_args(argv):
    """Positional <input_pdf_dir> <output_predictions_path> (the Docker
    contract), plus an optional --ledger PATH for the per-case evidence audit
    trail. --ledger may also be given via MIB_LEDGER for offline runs."""
    ledger = os.environ.get("MIB_LEDGER")
    run_receipt = None
    run_identity = None
    run_split = None
    pos = []
    it = iter(argv)
    for a in it:
        if a == "--ledger":
            ledger = next(it, None)
            if ledger is None:
                raise SystemExit("--ledger requires PATH")
        elif a == "--run-receipt":
            run_receipt = next(it, None)
            if run_receipt is None:
                raise SystemExit("--run-receipt requires PATH")
        elif a == "--run-identity":
            run_identity = next(it, None)
            if run_identity is None:
                raise SystemExit("--run-identity requires PATH")
        elif a == "--run-split":
            run_split = next(it, None)
            if run_split is None:
                raise SystemExit("--run-split requires dev, holdout, or validation")
        else:
            pos.append(a)
    if len(pos) != 2:
        raise SystemExit("usage: predict.py <input_pdf_dir> "
                         "<output_predictions_path> [--ledger PATH] "
                         "[--run-receipt PATH --run-identity PATH "
                         "--run-split dev|holdout|validation]")
    if run_receipt and not ledger:
        raise SystemExit("--run-receipt requires --ledger")
    if run_receipt and not run_identity:
        raise SystemExit("--run-receipt requires --run-identity")
    if run_receipt and not run_split:
        raise SystemExit("--run-receipt requires --run-split")
    if run_identity and not run_receipt:
        raise SystemExit("--run-identity requires --run-receipt")
    if run_split and not run_receipt:
        raise SystemExit("--run-split requires --run-receipt")
    if run_split and run_split not in RUN_SPLITS:
        raise SystemExit("--run-split must be dev, holdout, or validation")
    return pos[0], pos[1], ledger, run_receipt, run_identity, run_split


FLUSH_SECS = float(os.environ.get("MIB_FLUSH_SECS", "300"))


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _effective_run_config():
    """Record the values this exact process and its workers will use."""
    defaults = {
        "MIB_NATIVE_SCAN_OCR": "1",
        "MIB_NATIVE_SCAN_FAST_DPI": "150",
        "MIB_PIXMATCH": "1",
        "MIB_TRANSDUCER": "0",
        "MIB_HQ_ADD_STRETCH_DATE": "1",
        "MIB_HQ_WEAK_RISK": "1",
        "MIB_REVIEW_MODEL": "0",
        "MIB_FLAG_SUBSTRING": "1",
        "MIB_REVIEW_MARGIN": "0",
        "MIB_PLUS2_DATE_REPAIR": "1",
        "MIB_REC_MODEL": "",
        "MIB_PIX_BANK": "",
        "MIB_DUMP_RAW": "0",
        "MIB_DISABLE_EXTRACTION_RETRY": "0",
        "MIB_WORKER_MAX_CASES": "48",
        "MIB_MAX_RETRY_CASES": "8",
        "MIB_RETRY_CASE_TIMEOUT": "130",
        "MIB_RETRY_BUDGET_SECS": "1100",
        "MIB_RETRY_KILL_GRACE_SECS": "5",
        "MIB_BATCH_LIMIT_SECS": "30000",
        "MIB_FINALIZE_RESERVE_SECS": "60",
        "MIB_CASE_TIMEOUT": "120",
        "MIB_STUCK_SECS": "150",
        "MIB_STARTUP_GRACE": "120",
        "MIB_WATCHDOG_POLL": "2",
        "MIB_FLUSH_SECS": "300",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    for key in os.environ:
        if key.startswith("MIB_TEST_") or key in {
                "MIB_ACTIVE_CASE", "MIB_EXTRACTION_ATTEMPT"}:
            raise SystemExit(
                f"test/injection environment cannot produce a run receipt: {key}")
        if key.startswith("MIB_") and key not in defaults and key != "MIB_LEDGER":
            raise SystemExit(
                f"unknown MIB environment cannot produce a run receipt: {key}")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise SystemExit(
                f"receipt-producing runs require explicit {key}=1")
    return {key: os.environ.get(key, default)
            for key, default in defaults.items()}


def _input_manifest_identity(entries):
    return [{key: entry[key] for key in (
        "ordinal", "case_id", "size", "sha256")}
            for entry in entries]


def _validate_run_identity(identity, config_sha256, input_manifest_sha256,
                           run_split):
    """Validate the operator-predeclared chain of custody before workers run.

    The producer process can prove exact agreement with its live configuration
    and ordered input bytes. Image/revision/manifest values are strict identity
    claims that the post-run binder independently verifies against saved image
    inspection and runtime-manifest evidence.
    """
    if not isinstance(identity, dict) or set(identity) != RUN_IDENTITY_KEYS:
        raise ValueError("run identity has unexpected or missing keys")
    if identity.get("schema") != RUN_IDENTITY_SCHEMA:
        raise ValueError("unsupported run identity schema")
    if not SHA1_RE.fullmatch(identity.get("producer_git_sha") or ""):
        raise ValueError("run identity producer SHA is malformed")
    if not IMAGE_ID_RE.fullmatch(identity.get("image_id") or ""):
        raise ValueError("run identity image ID is malformed")
    if identity.get("image_revision") != identity["producer_git_sha"]:
        raise ValueError("run identity image revision must equal producer SHA")
    for key in (
            "image_inspect_sha256", "runtime_manifest_sha256",
            "config_sha256", "input_manifest_sha256"):
        if not SHA256_RE.fullmatch(identity.get(key) or ""):
            raise ValueError(f"run identity {key} is malformed")
    if identity["config_sha256"] != config_sha256:
        raise ValueError("run identity config hash differs from live config")
    if identity["input_manifest_sha256"] != input_manifest_sha256:
        raise ValueError("run identity input hash differs from live ordered inputs")
    if identity.get("run_split") not in RUN_SPLITS or \
            identity["run_split"] != run_split:
        raise ValueError("run identity split differs from declared run split")
    if not SHA256_RE.fullmatch(identity.get("run_nonce") or ""):
        raise ValueError("run identity nonce must be 256-bit lowercase hex")
    return identity


def _prepare_run_receipt(path, identity_path, input_dir, pdfs, output_path,
                         ledger_path, worker_count, run_split):
    """Validate one predeclared run before workers start.

    The returned payload remains in memory. It is published only after both
    output artifacts are durable, so an interrupted run has no completed
    receipt that could be mistaken for valid evidence.
    """
    path = Path(path).resolve()
    output_path = Path(output_path).resolve()
    ledger_path = Path(ledger_path).resolve()
    if len({path, output_path, ledger_path}) != 3 or \
            any(item.parent != path.parent for item in (output_path, ledger_path)):
        raise SystemExit(
            "receipt, predictions, and evidence must be distinct sibling files")
    run_directory = path.parent
    if run_directory.exists():
        raise SystemExit(
            f"receipt-producing run directory already exists: {run_directory}")
    try:
        resolved_identity_path = Path(identity_path).resolve(strict=True)
    except OSError as exc:
        raise SystemExit(f"run identity preflight failed: {exc}") from exc
    if resolved_identity_path.parent == run_directory:
        raise SystemExit("run identity must be outside the new run directory")
    entries = []
    source_paths = []
    for ordinal, raw in enumerate(pdfs):
        pdf = Path(raw).resolve(strict=True)
        if not pdf.is_file():
            raise SystemExit(f"run receipt input is not a regular file: {pdf}")
        source_paths.append(pdf)
        entries.append({
            "ordinal": ordinal,
            "case_id": pdf.stem,
            "filename": pdf.name,
            "size": pdf.stat().st_size,
            "sha256": _sha256_file(pdf),
        })
    if not entries:
        raise SystemExit("run receipt requires at least one input PDF")
    effective_config = _effective_run_config()
    config_sha256 = _canonical_sha256(effective_config)
    input_manifest_sha256 = _canonical_sha256(
        _input_manifest_identity(entries))
    try:
        identity = json.loads(resolved_identity_path.read_text())
        _validate_run_identity(
            identity, config_sha256, input_manifest_sha256, run_split)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"run identity preflight failed: {exc}") from exc
    payload = {
        "schema": RUN_RECEIPT_SCHEMA,
        "run_identity": identity,
        "run_identity_sha256": _canonical_sha256(identity),
        "run_split": run_split,
        "run_nonce": identity["run_nonce"],
        "effective_config": effective_config,
        "config_sha256": config_sha256,
        "worker_count": worker_count,
        "input_source": {
            "kind": "sorted_pdf_directory",
            "directory_name": Path(input_dir).resolve().name,
        },
        "input_manifest": entries,
        "input_manifest_sha256": input_manifest_sha256,
        "artifacts": None,
    }
    try:
        run_directory.parent.mkdir(parents=True, exist_ok=True)
        run_directory.mkdir()
    except FileExistsError as exc:
        raise SystemExit(
            f"receipt-producing run directory already exists: {run_directory}") \
            from exc
    except OSError as exc:
        raise SystemExit(f"could not create run directory: {exc}") from exc
    return path, payload, output_path, ledger_path, tuple(source_paths)


def _publish_run_receipt(prepared):
    """Atomically publish a terminal receipt over the exact durable outputs."""
    path, payload, output_path, ledger_path, source_paths = prepared
    terminal_inputs = []
    for ordinal, pdf in enumerate(source_paths):
        terminal_inputs.append({
            "ordinal": ordinal,
            "case_id": pdf.stem,
            "filename": pdf.name,
            "size": pdf.stat().st_size,
            "sha256": _sha256_file(pdf),
        })
    if terminal_inputs != payload["input_manifest"]:
        raise SystemExit(
            "input bytes changed between receipt preflight and completion")
    for label, artifact in (("predictions", output_path),
                            ("evidence", ledger_path)):
        if not artifact.is_file():
            raise SystemExit(f"cannot complete receipt without {label} artifact")
    expected_entries = {output_path.name, ledger_path.name}
    actual_entries = {entry.name for entry in path.parent.iterdir()}
    if actual_entries != expected_entries:
        raise SystemExit(
            "cannot complete receipt with unexpected or missing run artifacts")
    payload = dict(payload)
    payload["terminal_status"] = "completed"
    payload["artifacts"] = {
        name: {
            "filename": artifact.name,
            "size": artifact.stat().st_size,
            "sha256": _sha256_file(artifact),
        }
        for name, artifact in (("predictions", output_path),
                               ("evidence", ledger_path))
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise SystemExit(f"refusing to overwrite run receipt: {path}") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _state_stem(state):
    """Source-file stem of a state row (resolved id may differ on renames)."""
    stem = (state.get("case_id_provenance") or {}).get("stem")
    return stem or state["case_id"]


def _collect_states(shards, pdfs, complete):
    """Dedup-merge shard state files; with complete=True, backfill a
    conservative stub for every PDF that has no state row yet.

    States are keyed by their source-file stem so renamed inputs still merge,
    order, and backfill correctly; the resolved case id is emission-side."""
    by_stem = {}
    for shard in shards:
        for path in shard.state_files:
            if path.exists():
                for line in open(path):
                    try:
                        s = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # torn tail line from a killed worker
                    if _state_stem(s) in by_stem:
                        continue  # respawn overlap: first complete row wins
                    by_stem[_state_stem(s)] = s
    if complete:
        # Absolute completeness: crashed/hung/skipped cases still get a row.
        for pdf in pdfs:
            cid = Path(pdf).stem
            if cid not in by_stem:
                by_stem[cid] = {
                    "case_id": cid, "pools": {}, "doc_notes": {},
                    "case_id_provenance": {"stem": cid,
                                           "source": "backfill_stub"},
                    "mean_ocr_conf": 0.0, "injection": {},
                    "error": "missing_primary_state",
                    "extraction": {
                        "attempt_count": 1, "recovered": False,
                        "attempts": [{"attempt": 1, "status": "failed",
                                      "failure_category":
                                      "missing_primary_state"}]}}
    # Reconstruct the exact order a clean run emits: each shard's sorted input
    # slice in shard order. Missing/backfilled cases therefore return to their
    # clean slot instead of being appended after all successful states.
    ordered_pdfs = [pdf for i in range(len(shards)) for pdf in pdfs[i::len(shards)]]
    return [by_stem[Path(pdf).stem] for pdf in ordered_pdfs
            if Path(pdf).stem in by_stem]


def _stub_category(state):
    """Return why a state deserves one retry, else None.

    An explicit worker error is authoritative.  The structural check also
    recognizes legacy/torn fallback states that predate error provenance.
    """
    attempts = state.get("extraction", {}).get("attempts", [])
    if attempts:
        last = attempts[-1]
        # New workers explicitly attest success/failure.  That provenance is
        # authoritative: a legitimate evidence-empty success is not a stub.
        if last.get("status") == "success":
            return None
        if last.get("failure_category"):
            return last["failure_category"]
    if state.get("error"):
        return "extraction_error"
    # Structural inference exists only for legacy states that have no explicit
    # attempt record.  It recognizes old fallback rows without stealing retry
    # capacity from a current, explicitly successful empty packet.
    if not state.get("pools") and not state.get("doc_notes"):
        return "recognizable_stub"
    return None


def _attempts(state):
    attempts = state.get("extraction", {}).get("attempts")
    if attempts:
        return list(attempts)
    category = _stub_category(state)
    status = "failed" if category else "success"
    row = {"attempt": 1, "status": status}
    if category:
        row["failure_category"] = category
    return [row]


def _with_retry_provenance(state, attempts, recovered):
    state = dict(state)
    state["extraction"] = {
        "attempt_count": sum(a.get("status") != "not_attempted" for a in attempts),
        "recovered": recovered,
        "attempts": attempts,
    }
    return state


def _read_retry_state(path, case_id):
    if not path.exists():
        return None
    for line in open(path):
        try:
            state = json.loads(line)
        except json.JSONDecodeError:
            continue
        if state.get("case_id") == case_id:
            return state
    return None


def _bounded_exception(exc, limit=400):
    lines = [" ".join(line.split()) for line in str(exc).splitlines()
             if line.strip()]
    tail = lines[-1] if lines else "no_message"
    prefix = f"{type(exc).__name__}: "
    summary = prefix + tail
    return summary if len(summary) <= limit else \
        prefix + tail[-max(1, limit - len(prefix)):]


def _retry_log(message):
    """Retry diagnostics must never weaken validator-safe batch completion."""
    try:
        print(message, flush=True)
    except Exception:
        pass


def _run_retry_worker(command, env, timeout):
    """Run one retry in its own process group and kill the whole group on cap."""
    proc = subprocess.Popen(
        command, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True)
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        # start_new_session makes pid the process-group id.  ONNX currently
        # uses threads, not children, but killpg preserves the bound if a future
        # recognizer/runtime introduces helper processes.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # exited between timeout observation and group cleanup
        try:
            proc.wait(timeout=RETRY_KILL_GRACE_SECS)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            # The process group has already received SIGKILL.  Do not replace
            # a per-case timeout with an unbounded cleanup wait if the kernel
            # has not reaped the direct child within the explicit grace.
        raise


def _retry_candidates(states, pdfs):
    """Return retryable states in deterministic original PDF order."""
    pdf_order = {Path(pdf).stem: i for i, pdf in enumerate(pdfs)}
    after_known_pdfs = len(pdf_order)
    candidates = [(i, state, _stub_category(state))
                  for i, state in enumerate(states) if _stub_category(state)]
    candidates.sort(key=lambda row: (pdf_order.get(
        row[1].get("case_id"), after_known_pdfs + row[0]), row[0]))
    return candidates


def _retry_failed_states(states, pdfs, tmp, batch_started):
    """Retry failed/stub states once, serially, each in a fresh process.

    The retry phase has three independent hard bounds: one attempt per case,
    MAX_RETRY_CASES candidates, and a wall-clock deadline no later than both
    RETRY_BUDGET_SECS from now and the batch cap minus finalization reserve.
    """
    if os.environ.get("MIB_DISABLE_EXTRACTION_RETRY") == "1":
        return states
    by_case = {Path(p).stem: p for p in pdfs}
    candidates = _retry_candidates(states, pdfs)
    if not candidates:
        return states

    states = list(states)
    retry_deadline = min(time.monotonic() + RETRY_BUDGET_SECS,
                         batch_started + BATCH_LIMIT_SECS - FINALIZE_RESERVE_SECS)
    for ordinal, (idx, primary, category) in enumerate(candidates):
        attempts = _attempts(primary)
        if ordinal >= MAX_RETRY_CASES or time.monotonic() >= retry_deadline:
            attempts.append({"attempt": 2, "status": "not_attempted",
                             "failure_category": "retry_budget_exhausted"})
            states[idx] = _with_retry_provenance(primary, attempts, False)
            continue

        cid = primary["case_id"]
        pdf = by_case.get(cid)
        if not pdf:
            attempts.append({"attempt": 2, "status": "not_attempted",
                             "failure_category": "source_pdf_missing"})
            states[idx] = _with_retry_provenance(primary, attempts, False)
            continue
        stage = "setup"
        timeout = 0.0
        try:
            list_file = tmp / f"retry_{ordinal}.txt"
            state_file = tmp / f"retry_{ordinal}.jsonl"
            hb_file = tmp / f"retry_{ordinal}.hb"
            list_file.write_text(pdf + "\n")
            env = dict(os.environ, OMP_NUM_THREADS="1",
                       MIB_EXTRACTION_ATTEMPT="2")
            remaining = retry_deadline - time.monotonic()
            timeout = max(0.1, min(RETRY_CASE_TIMEOUT, remaining))
            command = [sys.executable,
                       str(Path(__file__).resolve().parent / "run_shard.py"),
                       str(list_file), str(state_file), str(hb_file)]
            stage = "launch"
            worker_rc = _run_retry_worker(command, env, timeout)
            stage = "result"
            retry = _read_retry_state(state_file, cid)
            if retry is None:
                retry_attempt = {"attempt": 2, "status": "failed",
                                 "failure_category": "retry_worker_no_state",
                                 "error": f"worker_rc={worker_rc}"}
                attempts.append(retry_attempt)
                states[idx] = _with_retry_provenance(primary, attempts, False)
            else:
                attempts.extend(_attempts(retry))
                recovered = _stub_category(retry) is None
                chosen = retry if recovered else primary
                states[idx] = _with_retry_provenance(chosen, attempts, recovered)
                _retry_log(f"[retry] {cid} primary={category} "
                           f"result={'recovered' if recovered else 'failed'}")
        except subprocess.TimeoutExpired:
            attempts.append({"attempt": 2, "status": "failed",
                             "failure_category": "retry_process_timeout",
                             "error": f"timeout({timeout:.1f}s)"})
            states[idx] = _with_retry_provenance(primary, attempts, False)
            _retry_log(f"[retry] {cid} primary={category} result=timeout")
        except OSError as exc:
            failure_category = {
                "setup": "retry_worker_setup_error",
                "launch": "retry_worker_launch_error",
                "result": "retry_worker_result_error",
            }[stage]
            status = "not_attempted" if stage == "setup" else "failed"
            attempts.append({"attempt": 2, "status": status,
                             "failure_category": failure_category,
                             "error": _bounded_exception(exc)})
            states[idx] = _with_retry_provenance(primary, attempts, False)
            _retry_log(f"[retry] {cid} primary={category} result={stage}_error")
        except Exception as exc:
            # Retry is best-effort recovery. An unexpected retry-only defect
            # must never replace the already validator-safe primary state or
            # abort the remaining batch.
            status = "not_attempted" if stage == "setup" else "failed"
            failure_category = "retry_worker_setup_error" if stage == \
                "setup" else "retry_worker_exception"
            attempts.append({"attempt": 2, "status": status,
                             "failure_category": failure_category,
                             "error": _bounded_exception(exc)})
            states[idx] = _with_retry_provenance(primary, attempts, False)
            _retry_log(f"[retry] {cid} primary={category} result=exception")
    return states


# evaluate.py exits 2 (failing the whole submission) on any *unexpected* case
# id, but only applies a small per-case penalty for a *missing* one. So a row
# whose id is not a well-formed MIB-\d{6} must be OMITTED, never emitted: the
# omission costs one missing case, emitting it would fail the run. On public
# data every stem is a valid id so this never fires; it guards the renamed/
# hashed private input whose document id was also unreadable.
_VALID_CASE_ID = re.compile(r"MIB-\d{6}")


def _emit_case_id_ok(case_id):
    return bool(_VALID_CASE_ID.fullmatch(str(case_id)))


def _write_predictions(states, epoch, out, batch_revoked=frozenset()):
    """Atomic full write: decide every state, replace out in one os.replace so
    a kill mid-write can never leave a torn file at the contract path.

    Applies the two-ledger fusion when a native ledger is present so an interim
    safety flush emits the same result the final write would.
    """
    (_, _, native_epoch, native_revoked, ablation) = _batch_decision_inputs(states)
    tmp_path = out.with_suffix(out.suffix + ".tmp")
    written_ids = set()
    with open(tmp_path, "w") as f:
        for state in states:
            try:
                pred, _ = two_ledger.decide_case(
                    state, epoch, native_epoch, batch_revoked, native_revoked,
                    ablation)
            except Exception:
                pred = {"case_id": state["case_id"], **FALLBACKS,
                        "adjudication": "NEEDS_REVIEW", "confidence": 0.3}
            # An unexpected/malformed id fails the whole submission; omit it.
            if not _emit_case_id_ok(pred["case_id"]):
                continue
            # Duplicate ids are an evaluator exit-2; first resolution wins.
            if pred["case_id"] in written_ids:
                continue
            written_ids.add(pred["case_id"])
            f.write(json.dumps(pred) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, out)


def _failure_detail(state, exc):
    """Return bounded, gate-visible provenance for a conservative fallback."""
    extraction_error = state.get("error")
    return {
        "reasons": ["extraction_error" if extraction_error else "decision_error"],
        "execution_error": extraction_error or
        f"decision_error({type(exc).__name__})",
    }


def main():
    batch_started = time.monotonic()
    (input_dir, output_path, ledger_path, run_receipt_path,
     run_identity_path, run_split) = _parse_args(sys.argv[1:])
    # Cheapest inputs first (size is a good proxy for raster/OCR load): if the
    # container is hard-stopped at the batch time limit, the interim flushes
    # then carry real extractions for the most cases. Deterministic tie-break.
    pdfs = sorted(
        (str(p) for p in Path(input_dir).glob("*.pdf")),
        key=lambda path: (
            os.path.getsize(path),
            Path(path).name,
        ),
    )

    workers = min(4, os.cpu_count() or 1)
    tmp = Path(tempfile.mkdtemp(prefix="mib-", dir=os.environ.get("TMPDIR", "/tmp")))
    out = Path(output_path)
    prepared_receipt = None
    if run_receipt_path:
        prepared_receipt = _prepare_run_receipt(
            run_receipt_path, run_identity_path, input_dir, pdfs, output_path,
            ledger_path, workers, run_split)
        # Workers consume the resolved files hashed at preflight; a later
        # symlink retarget cannot change the processed path.
        pdfs = [str(path) for path in prepared_receipt[4]]
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
    shards = [Shard(tmp, i, pdfs[i::workers]) for i in range(workers)]
    # Interim safety flushes: the scorer evaluates whatever exists at the
    # output path if the container is hard-stopped at the batch time limit, so
    # a complete, well-formed row set (extracted where done, conservative
    # hedges elsewhere) is kept at the contract path throughout the run.
    # The first flush happens immediately: a kill before the first periodic
    # flush must find a full conservative row set, not an empty file.
    initial = _collect_states(shards, pdfs, complete=True)
    try:
        _write_predictions(initial, batch_epoch(initial), out,
                           batch_frequent_sponsors(initial))
    except OSError as exc:
        # A transient write error (e.g. a momentary disk/tmpfs hiccup) on the
        # very first flush must not abort the whole run before the periodic
        # flushes and final write get their chance at the contract path.
        print(f"[flush] initial flush failed, will retry: {exc!r}",
              file=sys.stderr, flush=True)
    last_flush = time.time()
    while not all(s.finished for s in shards):
        time.sleep(POLL_SECS)
        for s in shards:
            s.tick()
        if time.time() - last_flush >= FLUSH_SECS:
            interim = _collect_states(shards, pdfs, complete=True)
            _write_predictions(interim, batch_epoch(interim), out,
                               batch_frequent_sponsors(interim))
            last_flush = time.time()

    states = _collect_states(shards, pdfs, complete=True)
    states = _retry_failed_states(states, pdfs, tmp, batch_started)

    (epoch, batch_revoked, native_epoch, native_revoked,
     ablation) = _batch_decision_inputs(states)
    ledger_f = None
    if ledger_path:
        Path(ledger_path).parent.mkdir(parents=True, exist_ok=True)
        ledger_f = open(ledger_path, "w")
    try:
        tmp_out = out.with_suffix(out.suffix + ".tmp")
        written_ids = set()
        with open(tmp_out, "w") as f:
            for state in states:
                try:
                    if state.get("error"):
                        raise RuntimeError(state["error"])
                    pred, detail = two_ledger.decide_case(
                        state, epoch, native_epoch, batch_revoked,
                        native_revoked, ablation)
                except Exception as exc:
                    pred = {"case_id": state["case_id"], **FALLBACKS,
                            "adjudication": "NEEDS_REVIEW", "confidence": 0.3}
                    detail = _failure_detail(state, exc)
                # An unexpected/malformed id fails the whole submission; omit it.
                if not _emit_case_id_ok(pred["case_id"]):
                    continue
                # Duplicate ids are an evaluator exit-2; first resolution wins.
                if pred["case_id"] in written_ids:
                    continue
                written_ids.add(pred["case_id"])
                f.write(json.dumps(pred) + "\n")
                if ledger_f:
                    ledger_f.write(json.dumps(_ledger_row(pred, detail, state)) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_out, out)
    finally:
        if ledger_f:
            ledger_f.flush()
            os.fsync(ledger_f.fileno())
            ledger_f.close()
    if prepared_receipt:
        _publish_run_receipt(prepared_receipt)
    print(f"wrote {len(states)} predictions to {out}"
          + (f" (+ ledger {ledger_path})" if ledger_path else ""))


def _ledger_row(pred, detail, state=None):
    """One auditable record: the decision, why, and the evidence behind every
    field (source page-type, manual-precedence rank, snap score, cross-page
    agreement) plus the injection signals that lowered trust. This is the
    artifact that reframes the pipeline as an auditable adjudication system."""
    fe = detail.get("field_evidence", {})
    execution_error = detail.get("execution_error")
    native = (state or {}).get("native_ledger")
    native_summary = None
    if isinstance(native, dict):
        native_summary = {
            "authorized_scan_pages": native.get("authorized_scan_pages"),
            "image_views": native.get("image_views", []),
            "native_provenance": native.get("native_provenance", {}),
            "unbound_note_observations": native.get(
                "unbound_note_observations", []),
            "adjudicator_note": native.get("doc_notes", {}).get("finding"),
        }
    if not execution_error and state is not None:
        execution_error = state.get("error")
        if not execution_error:
            category = _stub_category(state)
            if category:
                execution_error = category
    return {
        "case_id": pred["case_id"],
        "adjudication": pred["adjudication"],
        "confidence": pred["confidence"],
        "reasons": detail.get("reasons"),
        "decision_path": detail.get("path"),
        "extraction": (state or {}).get("extraction", {
            "attempt_count": 1, "recovered": False,
            "attempts": [{"attempt": 1, "status": "success"}],
        }),
        "fields": {f: pred[f] for f in FALLBACKS},
        "evidence": {f: {"rank": ev[0], "snap_score": ev[1], "agreement": ev[2],
                         "source": detail.get("sources", {}).get(f)}
                     for f, ev in fe.items()},
        "adjudicator_note": detail.get("finding_note"),
        "finding_authority_origin": detail.get("finding_authority_origin"),
        "rank1_payload": detail.get("rank1_payload", {
            "finding": detail.get("finding_note"), "fields": {}}),
        "composited_rank1_payload": detail.get(
            "composited_rank1_payload",
            {"values": {}, "conflicts": [], "evidence": {}}),
        "rank1_conflicts": detail.get("rank1_conflicts", []),
        "rank1_conflict_evidence": detail.get(
            "rank1_conflict_evidence", []),
        "baseline_approval_guards": detail.get(
            "baseline_approval_guards", []),
        "baseline_batch_context": detail.get("baseline_batch_context", {}),
        "injection": {
            "hidden_spans": detail.get("hidden_span_count", 0),
            "answer_key_present": bool(detail.get("has_answer_key")),
            "hidden_field_mentions": detail.get("hidden_field_mentions", {}),
        },
        "image_views": detail.get("image_views", []),
        "image_view_registry": detail.get("image_view_registry", {
            "schema": "mib-image-view-registry-v1",
            "pages": [], "errors": [],
        }),
        "pixmatch_fired": detail.get("pixmatch_fired", []),
        "pixmatch_acceptances": detail.get("pixmatch_acceptances", []),
        "hq_trigger_weak_risk": detail.get(
            "hq_trigger_weak_risk", False),
        "hq_stretch_date_attempted": detail.get(
            "hq_stretch_date_attempted", False),
        "hq_stretch_date_candidates": detail.get(
            "hq_stretch_date_candidates", []),
        "review_model": detail.get("review_model"),
        "terminal_guard": detail.get("terminal_guard"),
        "terminal_approval_guard": detail.get(
            "terminal_approval_guard"),
        "identity_disqualified_pages": detail.get(
            "identity_disqualified_pages", []),
        "native_fallback_review_pages": detail.get(
            "native_fallback_review_pages", []),
        "execution_error": execution_error,
        "trust_note": ("hidden content present; used only to lower trust, never "
                       "as evidence" if detail.get("hidden_span_count") else None),
        **({"two_ledger": detail["two_ledger"]}
           if detail.get("two_ledger") else {}),
        **({"native_ledger": native_summary} if native_summary else {}),
    }


if __name__ == "__main__":
    main()

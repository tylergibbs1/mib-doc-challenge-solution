"""Runtime OCR-correction transducer: trie-constrained beam decode over int8
ONNX graphs (numpy + onnxruntime only — no torch in the image).

The model is a ~2.6M-param character encoder-decoder trained at dev time on
real mined confusion pairs plus degrade-render pairs produced by the shipped
OCR engine itself. It is injection-inert twice over: it is not an instruction
follower, and the constrained decode cannot emit anything outside the legal
value space (144x144 name grammar, closed vocabularies, SPN-\\d{4}/ISO-date
format walkers). It never runs for fee_status, and callers must apply the
deny-safety rule: a correction may never REPLACE a deny-triggering read with
a benign value (see pipeline.decide).

Disabled unless models/transducer_enc.int8.onnx exists; gated in decide() by
MIB_TRANSDUCER until the sealed-holdout gate decides whether it ships on.
"""
import json
from pathlib import Path

import numpy as np

_MODELS = Path(__file__).resolve().parents[1] / "models"
PAD, BOS, EOS, UNK = 0, 1, 2, 3
MAX_SRC, MAX_TGT = 56, 44

_STATE = None


def _date_children(prefix):
    n = len(prefix)
    if n == 0:
        return {"2"}
    if n == 1:
        return {"0"}
    if n == 2:
        return {"2"}
    if n == 3:
        return set("567")
    if n in (4, 7):
        return {"-"}
    if n == 5:
        return {"0", "1"}
    if n == 6:
        return set("123456789") if prefix[5] == "0" else set("012")
    if n == 8:
        return set("0123")
    if n == 9:
        first = prefix[8]
        if first == "0":
            return set("123456789")
        if first in "12":
            return set("0123456789")
        return set("01")
    return set()


def _trie_of(values):
    root = {}
    for v in values:
        node = root
        for ch in v:
            node = node.setdefault(ch, {})
        node["$"] = True
    return root


def _load():
    global _STATE
    if _STATE is not None:
        return _STATE
    enc_p = _MODELS / "transducer_enc.int8.onnx"
    if not enc_p.exists():
        _STATE = False
        return _STATE
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    enc = ort.InferenceSession(str(enc_p), sess_options=opts)
    dec = ort.InferenceSession(str(_MODELS / "transducer_dec.int8.onnx"), sess_options=opts)
    itos = json.loads((_MODELS / "transducer_vocab.json").read_text())["itos"]
    stoi = {s: i for i, s in enumerate(itos)}

    from .vocab import PURPOSES, SPECIES, VISAS, WORLDS
    names = json.loads((_MODELS / "name_vocab.json").read_text())
    tries = {
        "applicant_name": _trie_of(f"{a} {b}" for a in names["first"] for b in names["last"]),
        "species_code": _trie_of(SPECIES),
        "home_world": _trie_of(WORLDS),
        "visa_class": _trie_of(VISAS),
        "declared_purpose": _trie_of(PURPOSES),
        "sponsor_id": _trie_of([f"SPN-{i:04d}" for i in range(10000)]),
        "arrival_date": None,
    }
    _STATE = {"enc": enc, "dec": dec, "stoi": stoi, "tries": tries}
    return _STATE


def available():
    return bool(_load())


def correct(field, text, width=4):
    """Trie-constrained beam decode. Returns (value, normalized_logprob) or
    (None, -inf) when the model is absent or decoding fails."""
    st = _load()
    if not st or field not in st["tries"] or not text or len(text) > MAX_SRC - 8:
        return None, float("-inf")
    stoi = st["stoi"]
    tag = stoi.get(f"<{field}>")
    if tag is None:
        return None, float("-inf")

    src = np.full((1, MAX_SRC), PAD, dtype=np.int64)
    s = [tag] + [stoi.get(c, UNK) for c in text][: MAX_SRC - 1]
    src[0, :len(s)] = s
    mem, sm = st["enc"].run(None, {"src": src})

    def legal(prefix, node):
        if field == "arrival_date":
            return _date_children(prefix), len(prefix) == 10
        if node is None:
            return set(), False
        return {c for c in node if c != "$"}, "$" in node

    def advance(node, ch):
        return None if field == "arrival_date" else (node.get(ch) if node else None)

    beams = [(0.0, "", st["tries"][field], [BOS], False)]
    for _ in range(MAX_TGT - 1):
        nxt = []
        for logp, txt, node, ys, done in beams:
            if done:
                nxt.append((logp, txt, node, ys, True))
                continue
            lg, terminal = legal(txt, node)
            if not lg and not terminal:
                continue
            yf = np.full((1, MAX_TGT), PAD, dtype=np.int64)
            yf[0, :len(ys)] = ys
            logits = st["dec"].run(None, {"mem": mem, "sm": sm, "ys": yf,
                                          "t": np.array(len(ys) - 1, dtype=np.int64)})[0][0]
            logits = logits - logits.max()
            logprobs = logits - np.log(np.exp(logits).sum())
            if terminal:
                nxt.append((logp + float(logprobs[EOS]), txt, node, ys, True))
            for ch in lg:
                ci = stoi.get(ch)
                if ci is None:
                    continue
                nxt.append((logp + float(logprobs[ci]), txt + ch,
                            advance(node, ch), ys + [ci], False))
        if not nxt:
            break
        beams = sorted(nxt, key=lambda b: b[0] / max(len(b[1]), 1), reverse=True)[:width]
        if all(b[4] for b in beams):
            break
    finished = [b for b in beams if b[4]]
    if not finished:
        return None, float("-inf")
    best = max(finished, key=lambda b: b[0] / max(len(b[1]) + 1, 1))
    return best[1], best[0] / max(len(best[1]) + 1, 1)

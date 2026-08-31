# tests/test_finetune_data.py
import json
import tempfile
from pathlib import Path
from voiceagent.finetune_data import synthesize_reply, prepare_finetune_data

def test_synthesize_reply_echoes_order_and_action():
    r = synthesize_reply(["ORD-5"], "order_status")
    assert "ORD-5" in r
    assert r.strip().endswith("ACTION: order_status")

def test_prepare_writes_valid_chat_jsonl():
    with tempfile.TemporaryDirectory() as d:
        csv_path = str(Path(d) / "in.csv")
        out = str(Path(d) / "out.jsonl")
        import csv
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id","language","intent","user_text","expected_action","key_facts","escalate","authenticated","amount"])
            w.writerow(["c1","en","refund","refund my order ORD-1","refund","ORD-1","false","true","1000"])
        n = prepare_finetune_data(csv_path, out)
        assert n == 1
        lines = Path(out).read_text().strip().splitlines()
        assert len(lines) == 1
        obj = json.loads(lines[0])
        roles = [m["role"] for m in obj["messages"]]
        assert roles == ["system", "user", "assistant"]
        assert "ORD-1" in obj["messages"][2]["content"]
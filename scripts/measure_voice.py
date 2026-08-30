import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voiceagent.voice import measure_voice_pipeline

if __name__ == "__main__":
    audio = sys.argv[1] if len(sys.argv) > 1 else None
    m = measure_voice_pipeline(audio)
    print(json.dumps(m, indent=2))
    Path("data/out/voice.json").write_text(json.dumps(m, indent=2))

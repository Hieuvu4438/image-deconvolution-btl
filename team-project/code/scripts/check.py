"""Run the local unittest gate and save its actual transcript."""
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone, timedelta

ROOT=Path(__file__).resolve().parents[2]

def main() -> None:
    stamp=datetime.now(timezone(timedelta(hours=7))).strftime('%Y%m%d_%H%M%S')
    output=ROOT/'outputs'/'verification'
    output.mkdir(parents=True,exist_ok=True)
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'code/tests'),'-v'],
                          capture_output=True,text=True,encoding='utf-8',errors='replace')
    transcript=result.stdout+result.stderr
    (output/f'unittest_{stamp}.log').write_text(transcript,encoding='utf-8')
    print(transcript)
    raise SystemExit(result.returncode)

if __name__=='__main__': main()

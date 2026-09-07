"""Despeja o schema real dos parquets crus.

Rode isso ANTES de confiar em metrics.py. Os nomes de coluna do FBref
mudam entre versoes do soccerdata, e a lista de candidatos em _col()
provavelmente precisa de ajuste na sua versao.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW  # noqa: E402


def main() -> None:
    files = sorted(RAW.glob("*.parquet"))
    if not files:
        print("nada em data/raw. rode: python src/ingest.py --full")
        return

    for f in files:
        df = pd.read_parquet(f)
        print(f"\n=== {f.name} | {len(df)} linhas ===")
        for c in df.columns:
            print(f"  {c:<38} {str(df[c].dtype):<10} ex: {_sample(df[c])}")


def _sample(s: pd.Series) -> str:
    nn = s.dropna()
    return str(nn.iloc[0])[:28] if len(nn) else "-"


if __name__ == "__main__":
    main()

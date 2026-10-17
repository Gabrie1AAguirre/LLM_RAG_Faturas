"""Carregamento e limpeza das faturas Nubank em CSV."""

from pathlib import Path

import pandas as pd


def parse_amount(x: str | float | int) -> float:
    """Converte o valor da fatura para float.

    Alguns CSVs já vêm como número (ponto decimal); outros como texto pt-BR ("- 2.345,67").
    """
    if isinstance(x, (int, float)):
        return float(x)
    x = "".join(ch for ch in str(x) if ch.isdigit() or ch in ",.-")  # tira espaços, R$, \xa0
    if "," in x:  # vírgula é o decimal; qualquer ponto é separador de milhar
        x = x.replace(".", "").replace(",", ".")
    return float(x)


def carregar_faturas(pasta: str | Path) -> pd.DataFrame:
    """Junta todos os CSVs `Nubank_AAAA-MM-DD.csv` da pasta num DataFrame limpo.

    Colunas: date, title, amount, fatura (data do arquivo), tipo ("gasto" ou "credito").
    """
    partes = []
    for arquivo in sorted(Path(pasta).glob("Nubank_*.csv")):
        parte = pd.read_csv(arquivo, dtype={"amount": str})
        parte["fatura"] = pd.Timestamp(arquivo.stem.removeprefix("Nubank_"))
        partes.append(parte)
    df = pd.concat(partes, ignore_index=True)

    df["date"] = pd.to_datetime(df["date"])
    df["title"] = df["title"].astype(str)
    df["amount"] = df["amount"].map(parse_amount).astype(float)

    # A mesma transação pode aparecer em duas faturas; já compras iguais
    # dentro de um mesmo arquivo (dois cafés no dia) são legítimas.
    df["n"] = df.groupby(["fatura", "date", "title", "amount"]).cumcount()
    df = df.drop_duplicates(subset=["date", "title", "amount", "n"]).drop(columns="n")

    df["tipo"] = df["amount"].map(lambda v: "gasto" if v > 0 else "credito")
    return df.sort_values("date", kind="stable").reset_index(drop=True)

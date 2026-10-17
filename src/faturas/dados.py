"""Carregamento e limpeza das faturas Nubank em CSV."""


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

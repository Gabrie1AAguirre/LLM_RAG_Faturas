import pytest

from faturas.dados import parse_amount


@pytest.mark.parametrize("entrada, esperado", [
    ("123,45", 123.45),
    ("- 2.345,67", -2345.67),
    ("1.234,56", 1234.56),
    ("R$\xa067,89", 45.72),
    (31.5, 31.5),
    (10, 10.0),
    ("12.50", 12.5),  # CSV que já vem com ponto decimal
])
def test_parse_amount(entrada, esperado):
    assert parse_amount(entrada) == pytest.approx(esperado)

import pandas as pd
import pytest

from faturas.dados import carregar_faturas, parse_amount


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


def escrever_csv(pasta, nome, linhas):
    (pasta / nome).write_text("date,title,amount\n" + "".join(l + "\n" for l in linhas), encoding="utf-8")


def test_carregar_junta_e_tipa(tmp_path):
    escrever_csv(tmp_path, "Nubank_2026-09-08.csv",
                 ['2026-09-01,Padaria,"10,00"', '2026-09-02,Pagamento recebido,"- 3.456,00"'])
    escrever_csv(tmp_path, "Nubank_2026-10-12.csv", ['2026-10-04,Pizzaria Exemplo,"123,45"'])
    df = carregar_faturas(tmp_path)
    assert list(df.columns) == ["date", "title", "amount", "fatura", "tipo"]
    assert len(df) == 3
    assert df["tipo"].tolist() == ["gasto", "credito", "gasto"]
    assert df.loc[2, "fatura"] == pd.Timestamp("2026-10-12")
    assert df.loc[1, "amount"] == -3456.0


def test_remove_duplicata_entre_arquivos(tmp_path):
    escrever_csv(tmp_path, "Nubank_2026-09-08.csv", ['2026-09-30,Pet Shop Exemplo,"67,89"'])
    escrever_csv(tmp_path, "Nubank_2026-10-12.csv", ['2026-09-30,Pet Shop Exemplo,"67,89"'])
    assert len(carregar_faturas(tmp_path)) == 1


def test_mantem_compras_iguais_no_mesmo_arquivo(tmp_path):
    escrever_csv(tmp_path, "Nubank_2026-10-12.csv", ['2026-10-01,Cafe,"5,00"', '2026-10-01,Cafe,"5,00"'])
    assert len(carregar_faturas(tmp_path)) == 2


def test_arquivo_so_com_cabecalho(tmp_path):
    escrever_csv(tmp_path, "Nubank_2026-09-08.csv", [])
    escrever_csv(tmp_path, "Nubank_2026-10-12.csv", ['2026-10-01,Cafe,"5,00"'])
    assert len(carregar_faturas(tmp_path)) == 1

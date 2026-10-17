from types import SimpleNamespace

import ollama
import pandas as pd
import pytest

from faturas.tools import criar_ferramentas, perguntar_com_tools


@pytest.fixture
def df():
    return pd.DataFrame({
        "date": pd.to_datetime(["2025-03-10", "2025-06-07", "2025-12-31", "2026-01-02"]),
        "title": ["Pizzaria Exemplo", "Pagamento recebido", "Padaria", "Padaria"],
        "amount": [123.45, -3456.0, 10.0, 20.0],
        "tipo": ["gasto", "credito", "gasto", "gasto"],
    })


@pytest.fixture
def ferramentas(df):
    return {f.__name__: f for f in criar_ferramentas(df)}


def resposta(content="", tool_calls=None):
    return SimpleNamespace(message=SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls))


def chamada(nome, **argumentos):
    return SimpleNamespace(function=SimpleNamespace(name=nome, arguments=argumentos))


def test_total_gasto_inclusivo_e_ignora_credito(ferramentas):
    assert ferramentas["total_gasto"]("2025-01-01", "2025-12-31") == 133.45


def test_gastos_por_mes(ferramentas):
    assert ferramentas["gastos_por_mes"]() == {"2025-03": 123.45, "2025-12": 10.0, "2026-01": 20.0}


def test_buscar_ignora_caixa_e_acento(ferramentas):
    assert ferramentas["buscar_estabelecimento"]("pizzaria exemplo")["quantidade"] == 1
    encontrados = ferramentas["buscar_estabelecimento"]("PADARÍA")
    assert encontrados == {
        "quantidade": 2,
        "total_gasto": 30.0,
        "transacoes": [
            {"date": "2026-01-02", "title": "Padaria", "amount": 20.0},
            {"date": "2025-12-31", "title": "Padaria", "amount": 10.0},
        ],
    }


def test_buscar_limita_transacoes_mas_soma_tudo():
    # o resultado volta para o prompt: centenas de linhas estourariam a janela de contexto
    muitos = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=300),
        "title": ["Mercado"] * 300,
        "amount": [1.0] * 300,
        "tipo": ["gasto"] * 300,
    })
    buscar = {f.__name__: f for f in criar_ferramentas(muitos)}["buscar_estabelecimento"]
    resultado = buscar("mercado")
    assert resultado["quantidade"] == 300
    assert resultado["total_gasto"] == 300.0
    assert len(resultado["transacoes"]) == 20
    assert resultado["transacoes"][0]["date"] == "2025-10-27"  # as 20 mais recentes


def test_buscar_nome_vazio_e_erro(ferramentas):
    assert "erro" in ferramentas["buscar_estabelecimento"]("  ")


def test_ferramentas_geram_schema_para_o_modelo(ferramentas):
    from ollama._utils import convert_function_to_tool

    schema = convert_function_to_tool(ferramentas["total_gasto"])
    assert schema.function.name == "total_gasto"
    assert set(schema.function.parameters.properties) == {"inicio", "fim"}


def test_loop_executa_tool_e_responde(monkeypatch, df):
    respostas = [resposta(tool_calls=[chamada("total_gasto", inicio="2025-01-01", fim="2025-12-31")]),
                 resposta("R$ 133,45")]
    vistas = []

    def chat(**kw):
        vistas.append(list(kw["messages"]))
        return respostas.pop(0)

    monkeypatch.setattr(ollama, "chat", chat)
    assert perguntar_com_tools("quanto gastei em 2025?", df) == "R$ 133,45"
    ultima = vistas[1][-1]
    assert ultima["role"] == "tool" and ultima["tool_name"] == "total_gasto" and ultima["content"] == "133.45"


def test_tool_desconhecida_nao_quebra(monkeypatch, df):
    respostas = [resposta(tool_calls=[chamada("inventada")]), resposta("desculpe")]
    vistas = []

    def chat(**kw):
        vistas.append(list(kw["messages"]))
        return respostas.pop(0)

    monkeypatch.setattr(ollama, "chat", chat)
    assert perguntar_com_tools("x", df) == "desculpe"
    assert "não existe" in vistas[1][-1]["content"]


def test_loop_usa_temperatura_zero(monkeypatch, df):
    # respostas repetíveis: a mesma pergunta deve escolher a mesma ferramenta
    opcoes = []

    def chat(**kw):
        opcoes.append(kw.get("options"))
        return resposta("ok")

    monkeypatch.setattr(ollama, "chat", chat)
    perguntar_com_tools("x", df)
    assert opcoes == [{"temperature": 0}]


def test_docstrings_orientam_a_escolha(ferramentas):
    assert "ano" in ferramentas["total_gasto"].__doc__
    assert "não some" in ferramentas["gastos_por_mes"].__doc__.lower()

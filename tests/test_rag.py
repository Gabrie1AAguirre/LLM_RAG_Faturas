from types import SimpleNamespace

import numpy as np
import ollama
import pandas as pd
import pytest

from faturas.rag import Indice, buscar, carregar_ou_criar_indice, embedar, transacao_para_texto


def falso_embed(vetores_por_texto, chamadas=None):
    """Substitui ollama.embed: devolve vetores fixos por texto (ignorando o prefixo)."""

    def embed(model, input):
        if chamadas is not None:
            chamadas.append(input)
        return SimpleNamespace(embeddings=[vetores_por_texto(t.split(": ", 1)[-1]) for t in input])

    return embed


def df_exemplo(n):
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-10-01", "2026-10-02", "2026-10-03"][:n]),
        "title": ["Padaria", "Posto", "Mercado"][:n],
        "amount": [10.0, 200.0, 55.5][:n],
        "tipo": ["gasto"] * n,
    })


def test_transacao_para_texto():
    linha = pd.Series({"date": pd.Timestamp("2026-10-04"), "title": "Pizzaria Exemplo",
                       "amount": 123.45, "tipo": "gasto"})
    assert transacao_para_texto(linha) == "2026-10-04 | Pizzaria Exemplo | R$ 123,45 | gasto"
    linha2 = linha.copy()
    linha2["amount"] = -2345.67
    linha2["tipo"] = "credito"
    assert transacao_para_texto(linha2).endswith("R$ -2.345,67 | credito")


def test_embedar_normaliza(monkeypatch):
    monkeypatch.setattr(ollama, "embed", falso_embed(lambda t: [3.0, 4.0]))
    np.testing.assert_allclose(embedar(["x"], "search_document: "), [[0.6, 0.8]])


def test_embedar_usa_prefixo(monkeypatch):
    chamadas = []
    monkeypatch.setattr(ollama, "embed", falso_embed(lambda t: [1.0, 0.0], chamadas))
    embedar(["padaria"], "search_query: ")
    assert chamadas == [["search_query: padaria"]]


def test_buscar_ordena_por_similaridade(monkeypatch):
    monkeypatch.setattr(ollama, "embed", falso_embed(lambda t: [1.0, 0.0]))
    vetores = np.array([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]])
    indice = Indice(textos=["A", "B", "C"], vetores=vetores)
    assert buscar("q", indice, k=2) == ["A", "C"]


def test_cache_invalida_quando_dados_mudam(tmp_path, monkeypatch):
    chamadas = []
    monkeypatch.setattr(ollama, "embed", falso_embed(lambda t: [1.0, float(len(t))], chamadas))
    cache = tmp_path / "embeddings.npz"

    carregar_ou_criar_indice(df_exemplo(2), cache)
    assert len(chamadas) == 1 and cache.exists()

    carregar_ou_criar_indice(df_exemplo(2), cache)
    assert len(chamadas) == 1  # usou o cache

    indice = carregar_ou_criar_indice(df_exemplo(3), cache)
    assert len(chamadas) == 2  # dados mudaram: recalculou
    assert len(indice.textos) == 3
    assert indice.vetores.shape == (3, 2)


def test_embedar_em_lotes(monkeypatch):
    chamadas = []
    monkeypatch.setattr(ollama, "embed", falso_embed(lambda t: [1.0, 0.0], chamadas))
    vetores = embedar([f"t{i}" for i in range(150)], "search_document: ", lote=64)
    assert [len(c) for c in chamadas] == [64, 64, 22]
    assert vetores.shape == (150, 2)


def test_embedar_repete_lote_que_falhou(monkeypatch):
    # o Ollama pode descarregar o modelo de embeddings no meio da chamada (VRAM pequena)
    respostas = [ollama.ResponseError("connection refused", 400), None]
    normal = falso_embed(lambda t: [1.0, 0.0])

    def embed(model, input):
        erro = respostas.pop(0) if respostas else None
        if erro:
            raise erro
        return normal(model, input)

    monkeypatch.setattr(ollama, "embed", embed)
    assert embedar(["x"], "search_document: ").shape == (1, 2)

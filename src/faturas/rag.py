"""RAG "na mão": transações viram texto, texto vira vetor, a pergunta busca os vetores mais próximos."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ollama
import pandas as pd

from faturas.llm import MODELO_EMBED, chamar_ollama, conversar

# O nomic-embed-text foi treinado com prefixos que dizem se o texto é documento ou pergunta.
PREFIXO_DOCUMENTO = "search_document: "
PREFIXO_PERGUNTA = "search_query: "

INSTRUCOES = (
    "Você é um assistente que responde perguntas sobre as faturas do cartão do usuário. "
    "Use SOMENTE as transações fornecidas. Se a resposta não estiver nelas, diga que não sabe. "
    "Valores positivos são gastos; negativos são pagamentos, estornos ou descontos. Responda em português."
)


@dataclass
class Indice:
    textos: list[str]
    vetores: np.ndarray


def _reais(valor: float) -> str:
    return f"{valor:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")


def transacao_para_texto(linha: pd.Series) -> str:
    return f"{linha['date']:%Y-%m-%d} | {linha['title']} | R$ {_reais(linha['amount'])} | {linha['tipo']}"


def _embedar_lote(textos: list[str], tentativas: int = 3) -> list[list[float]]:
    # Com pouca VRAM, o Ollama pode descarregar o modelo de embeddings no meio da
    # chamada para abrir espaço ao modelo de chat; a nova tentativa o recarrega.
    for tentativa in range(tentativas):
        try:
            return chamar_ollama(ollama.embed, modelo=MODELO_EMBED, model=MODELO_EMBED, input=textos).embeddings
        except ollama.ResponseError:
            if tentativa == tentativas - 1:
                raise


def embedar(textos: list[str], prefixo: str, lote: int = 64) -> np.ndarray:
    """Transforma textos em vetores de norma 1 (assim o produto escalar é a similaridade de cosseno)."""
    com_prefixo = [prefixo + t for t in textos]
    vetores = []
    for i in range(0, len(com_prefixo), lote):
        vetores.extend(_embedar_lote(com_prefixo[i:i + lote]))
    vetores = np.array(vetores, dtype=float)
    return vetores / np.linalg.norm(vetores, axis=1, keepdims=True)


def carregar_ou_criar_indice(df: pd.DataFrame, cache: Path) -> Indice:
    """Reaproveita os embeddings salvos se os textos forem os mesmos; senão recalcula."""
    textos = [transacao_para_texto(linha) for _, linha in df.iterrows()]
    cache = Path(cache)
    if cache.exists():
        salvo = np.load(cache)
        if salvo["textos"].tolist() == textos:
            return Indice(textos=textos, vetores=salvo["vetores"])
    vetores = embedar(textos, PREFIXO_DOCUMENTO)
    np.savez(cache, textos=np.array(textos), vetores=vetores)
    return Indice(textos=textos, vetores=vetores)


def buscar(pergunta: str, indice: Indice, k: int = 10) -> list[str]:
    similaridades = indice.vetores @ embedar([pergunta], PREFIXO_PERGUNTA)[0]
    melhores = np.argsort(-similaridades)[:k]
    return [indice.textos[i] for i in melhores]


def responder(pergunta: str, indice: Indice, k: int = 10) -> str:
    contexto = "\n".join(buscar(pergunta, indice, k))
    return conversar([
        {"role": "system", "content": INSTRUCOES},
        {"role": "user", "content": f"Transações:\n{contexto}\n\nPergunta: {pergunta}"},
    ])

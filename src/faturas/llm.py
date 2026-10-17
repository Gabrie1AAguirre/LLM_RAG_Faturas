"""Conversa com o modelo local via Ollama (http://localhost:11434)."""

import re

import httpx
import ollama

MODELO_CHAT = "qwen3:4b-instruct"  # sem raciocínio <think>: ~2x mais rápido que o qwen3:4b
MODELO_EMBED = "nomic-embed-text"


class OllamaIndisponivel(RuntimeError):
    """O Ollama não respondeu ou o modelo pedido não está baixado."""


def remover_think(texto: str) -> str:
    """Tira o bloco <think>...</think> que o qwen3 escreve antes da resposta.

    Na variante "thinking" o <think> de abertura fica no prompt, então a resposta
    pode trazer só o </think>: nesse caso tudo antes dele é raciocínio.
    """
    texto = re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL)
    return texto.rsplit("</think>", 1)[-1].strip()


def chamar_ollama(fn, *args, modelo: str, **kwargs):
    """Executa uma chamada ao Ollama trocando os erros técnicos por mensagens claras."""
    try:
        return fn(*args, **kwargs)
    except (ConnectionError, httpx.ConnectError) as e:
        raise OllamaIndisponivel("Não consegui falar com o Ollama. Abra o app Ollama e tente de novo.") from e
    except ollama.ResponseError as e:
        if e.status_code == 404:
            raise OllamaIndisponivel(f"Modelo '{modelo}' não encontrado. Rode: ollama pull {modelo}") from e
        raise


def conversar(mensagens: list[dict], modelo: str = MODELO_CHAT, **kwargs) -> str:
    """Envia as mensagens ao modelo e devolve só o texto da resposta."""
    resposta = chamar_ollama(ollama.chat, modelo=modelo, model=modelo, messages=mensagens, think=False, **kwargs)
    return remover_think(resposta.message.content)

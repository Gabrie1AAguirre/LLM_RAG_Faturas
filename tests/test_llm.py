import ollama
import pytest

from faturas.llm import OllamaIndisponivel, conversar, remover_think


def test_remover_think():
    assert remover_think("<think>\nblá\n</think>\n\nok") == "ok"
    assert remover_think("sem think") == "sem think"


def test_remover_think_sem_tag_de_abertura():
    # o template do qwen3 "thinking" já abre <think> no prompt; a resposta só traz o fechamento
    assert remover_think("Okay, the user wants...\n</think>\n\noi") == "oi"


def test_ollama_fechado(monkeypatch):
    def falha(**kw):
        raise ConnectionError("refused")

    monkeypatch.setattr(ollama, "chat", falha)
    with pytest.raises(OllamaIndisponivel, match="Abra o app Ollama"):
        conversar([{"role": "user", "content": "oi"}])


def test_modelo_nao_baixado(monkeypatch):
    def falha(**kw):
        raise ollama.ResponseError("model not found", 404)

    monkeypatch.setattr(ollama, "chat", falha)
    with pytest.raises(OllamaIndisponivel, match="ollama pull qwen3:4b-instruct$"):
        conversar([{"role": "user", "content": "oi"}])

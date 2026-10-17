"""Tool calling: o modelo escolhe uma função, o Python executa, o modelo redige a resposta."""

import json
import unicodedata
from collections.abc import Callable
from datetime import date

import ollama
import pandas as pd

from faturas.llm import MODELO_CHAT, chamar_ollama, remover_think

MAX_TRANSACOES = 20


def _normalizar(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in sem_acento if not unicodedata.combining(c)).lower()


def criar_ferramentas(df: pd.DataFrame) -> list[Callable]:
    """Cria as funções que o modelo pode chamar, todas olhando para o mesmo DataFrame.

    As docstrings e os type hints viram o "manual" (schema JSON) que o modelo lê.
    """
    gastos = df[df["tipo"] == "gasto"]

    def total_gasto(inicio: str, fim: str) -> float:
        """Total gasto no cartão entre duas datas, inclusive. Use para "quanto gastei em um ano/mês/período".

        Exemplo: o ano de 2025 inteiro é inicio="2025-01-01", fim="2025-12-31".

        Args:
            inicio: data inicial no formato AAAA-MM-DD
            fim: data final no formato AAAA-MM-DD

        Returns:
            Total gasto em reais
        """
        periodo = gastos[(gastos["date"] >= pd.Timestamp(inicio)) & (gastos["date"] <= pd.Timestamp(fim))]
        return round(float(periodo["amount"].sum()), 2)

    def gastos_por_mes() -> dict[str, float]:
        """Total gasto em cada mês, no formato {"AAAA-MM": total}. Use para comparar meses.

        Não some estes valores para obter o total de um período: use total_gasto.

        Returns:
            Dicionário mês -> total gasto em reais
        """
        mensal = gastos.groupby(gastos["date"].dt.strftime("%Y-%m"))["amount"].sum()
        return {mes: round(float(total), 2) for mes, total in mensal.items()}

    def buscar_estabelecimento(nome: str) -> dict:
        """Procura transações cujo estabelecimento contém o nome dado (ignora maiúsculas e acentos).

        Já devolve o total gasto calculado: use esse número, não some as transações.

        Args:
            nome: parte do nome do estabelecimento, por exemplo "pizzaria exemplo"

        Returns:
            Quantidade de transações, total gasto e as 20 transações mais recentes
        """
        alvo = _normalizar(nome).strip()
        if not alvo:
            return {"erro": "informe parte do nome do estabelecimento"}
        achados = df[df["title"].map(lambda t: alvo in _normalizar(t))]
        # O resultado volta para o prompt: limitar evita estourar a janela de contexto.
        recentes = achados.sort_values("date", ascending=False).head(MAX_TRANSACOES)
        return {
            "quantidade": len(achados),
            "total_gasto": round(float(achados.loc[achados["tipo"] == "gasto", "amount"].sum()), 2),
            "transacoes": [
                {"date": f"{l['date']:%Y-%m-%d}", "title": l["title"], "amount": float(l["amount"])}
                for _, l in recentes.iterrows()
            ],
        }

    return [total_gasto, gastos_por_mes, buscar_estabelecimento]


def _instrucoes() -> str:
    return (
        f"Hoje é {date.today():%Y-%m-%d}. Você responde perguntas sobre as faturas do cartão do usuário. "
        "Para QUALQUER número (totais, somas, meses), use as ferramentas; nunca calcule de cabeça. "
        "Responda em português, com valores no formato R$ 1.234,56."
    )


def perguntar_com_tools(pergunta: str, df: pd.DataFrame, max_rodadas: int = 5) -> str:
    ferramentas = criar_ferramentas(df)
    por_nome = {f.__name__: f for f in ferramentas}
    mensagens = [{"role": "system", "content": _instrucoes()}, {"role": "user", "content": pergunta}]

    for _ in range(max_rodadas):
        resposta = chamar_ollama(ollama.chat, modelo=MODELO_CHAT, model=MODELO_CHAT,
                                 messages=mensagens, tools=ferramentas, think=False,
                                 options={"temperature": 0})
        if not resposta.message.tool_calls:
            return remover_think(resposta.message.content)

        mensagens.append(resposta.message)
        for chamada in resposta.message.tool_calls:
            nome = chamada.function.name
            if nome not in por_nome:
                resultado = {"erro": f"ferramenta {nome} não existe"}
            else:
                try:
                    resultado = por_nome[nome](**chamada.function.arguments)
                except (TypeError, ValueError) as e:  # argumentos inválidos vindos do modelo
                    resultado = {"erro": str(e)}
            mensagens.append({"role": "tool", "tool_name": nome,
                              "content": json.dumps(resultado, ensure_ascii=False)})

    return "Não consegui chegar a uma resposta usando as ferramentas."

# LLM + RAG local sobre faturas de cartão

Projeto de estudo: um LLM rodando **na sua máquina** (via [Ollama](https://ollama.com)) responde perguntas
sobre faturas do cartão Nubank exportadas em CSV. Nenhum dado sai do computador.

A ideia é entender cada peça, sem frameworks: chat com LLM, embeddings, busca vetorial (RAG) e tool calling.

## O que tem aqui

| Notebook | O que mostra |
|---|---|
| `01_chat_sem_rag` | Conversar com o modelo; ele inventa respostas quando não tem os dados; por que não dá para colar tudo no prompt |
| `02_dados` | Limpar e juntar os CSVs (valores `"1.234,56"`, duplicatas entre faturas, gasto × crédito) |
| `03_rag` | Embeddings, similaridade de cosseno, busca e resposta — e onde o RAG quebra |
| `04_tools` | O modelo chama funções pandas para fazer contas exatas |

A principal lição: **RAG serve para *encontrar* informação; para *calcular* sobre muitos registros, use tools.**
Para "quanto gastei no ano?", o RAG vê só 10 transações e soma errado; com tools o número é exato.

O código reutilizável fica em `src/faturas/`:

- `dados.py`: `carregar_faturas(pasta)` junta e limpa os CSVs
- `llm.py`: conversa com o Ollama, com mensagens claras se ele estiver fechado ou faltar o modelo
- `rag.py`: RAG "na mão" com numpy (embeddings em lotes + cache em `data/embeddings.npz`)
- `tools.py`: ferramentas (`total_gasto`, `gastos_por_mes`, `buscar_estabelecimento`) e o loop de tool calling

## Como rodar

Pré-requisitos: [uv](https://docs.astral.sh/uv/) e [Ollama](https://ollama.com/download).

```bash
# modelos (testado numa GTX 1650 de 4 GB)
ollama pull qwen3:4b-instruct
ollama pull nomic-embed-text

# ambiente Python
uv sync
uv run nbstripout --install --attributes .gitattributes   # tira as saídas dos notebooks no commit

# testes (não precisam do Ollama)
uv run pytest
```

Coloque as faturas exportadas do app Nubank em `data/`, com o nome `Nubank_AAAA-MM-DD.csv` e o formato:

```csv
date,title,amount
2026-10-04,Pizzaria Exemplo,"123,45"
2026-10-08,Pagamento recebido,"- 2.345,67"
```

Depois abra os notebooks em ordem, com o app do Ollama aberto.

## Privacidade

Faturas são dados pessoais, então:

- `data/` está no `.gitignore` (CSVs e cache de embeddings nunca são versionados);
- o `nbstripout` remove as saídas dos notebooks antes de cada commit;
- `tests/test_privacidade.py` falha se alguma dessas proteções deixar de funcionar.

## Limitações conhecidas

- O `nomic-embed-text` é treinado principalmente em inglês e distingue mal significados em português
  (o notebook 03 mostra isso). Um modelo multilíngue como `bge-m3` é um bom próximo passo.
- Modelos de 4B erram: as docstrings das ferramentas e `temperature=0` foram necessários para ele escolher a ferramenta certa.
- Estornos não são descontados dos totais de gasto.

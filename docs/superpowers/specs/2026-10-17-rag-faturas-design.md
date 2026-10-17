# RAG local sobre faturas Nubank — Design

**Data:** 2026-10-17
**Status:** aguardando revisão

## Objetivo

Projeto de **aprendizado** de LLM e RAG usando as próprias faturas do cartão Nubank como dataset.
O foco é entender cada peça (embedding, busca, prompt, tool calling), não entregar um produto.

**Sucesso =** conseguir conversar em português com um modelo local sobre os gastos e saber explicar:
1. por que o modelo sem RAG inventa respostas;
2. como o RAG resolve perguntas de *busca* ("quando comprei no X?");
3. por que o RAG falha em perguntas de *agregação* ("quanto gastei em 2025?") e como tool calling resolve.

## Restrições

- **Tudo local**: dados financeiros pessoais não saem da máquina. Modelos via Ollama (`localhost:11434`).
- **Hardware**: GTX 1650, 4 GB VRAM. `qwen3:4b` roda a ~15 tokens/s com split 34% CPU / 66% GPU.
- **Python 3.14** com `uv`. Evitar libs que ainda não suportam 3.14.
- **Dados**: 22 CSVs `data/Nubank_AAAA-MM-DD.csv`, colunas `date,title,amount`, ~1.050 linhas no total.
  `amount` vem misturado: texto pt-BR (`"123,45"`, `"- 2.345,67"`) ou float. Negativos são
  pagamentos de fatura, estornos e descontos.
- Os dados (`data/`) **não** entram no git.

## Abordagem escolhida: RAG "na mão"

Cliente Python oficial `ollama` + `numpy`. Sem LangChain/LlamaIndex nem banco vetorial: ~1.050
transações cabem em memória e a similaridade de cosseno é uma linha de numpy. Isso deixa cada
etapa visível. Frameworks foram descartados porque escondem justamente o que se quer aprender.

**Modelos**
- Chat: `qwen3:4b` (já baixado). Alternativa mais leve se ficar lento: `llama3.2:3b`.
- Embeddings: `nomic-embed-text` (baixar na etapa 03).
- O `qwen3` emite raciocínio em `<think>`; desligar com `think=False` ou remover o bloco da resposta exibida.

## Estrutura

```
src/faturas/
  __init__.py
  dados.py        # carregar_faturas(pasta) -> DataFrame limpo
  rag.py          # criado na etapa 03
tests/
  test_dados.py
  test_rag.py     # criado na etapa 03
notebooks/
  EDA.ipynb                # existente, fica como está
  01_chat_sem_rag.ipynb
  02_dados.ipynb
  03_rag.ipynb
  04_tools.ipynb
```

Os notebooks experimentam e explicam com markdown didático em português. A lógica reutilizável
vive em `src/faturas/` e tem testes com `pytest`.

## Etapas

### 01 — Chat sem RAG
- Chamar `ollama.chat` com `qwen3:4b` e streaming.
- Perguntar sobre as faturas **sem dar dados** → observar a alucinação.
- Depois colar umas 20 transações direto no prompt → funciona, mas mostrar o limite: 1.050 linhas
  não cabem bem no contexto de 4096 tokens. É a motivação para o RAG.

### 02 — Dados (`dados.py`)
- `parse_amount(x) -> float`: reaproveita a lógica do `EDA.ipynb` (ponto = milhar, vírgula = decimal,
  remove espaço/R$/`\xa0`, aceita float).
- `carregar_faturas(pasta) -> DataFrame` com colunas `date` (datetime), `title` (str), `amount` (float),
  `fatura` (data extraída do nome do arquivo) e `tipo` (`"gasto"` se amount > 0, `"credito"` caso contrário).
- Remove linhas duplicadas exatas entre arquivos, se houver.
- Testes: `"123,45"` → 123.45; `"- 2.345,67"` → -2345.67; `"1.234,56"` → 1234.56; float passa direto;
  `carregar_faturas` em pasta temporária com 2 CSVs pequenos.

### 03 — RAG (`rag.py`)
- **Documento** = uma transação em texto: `"2026-10-04 | Pizzaria Exemplo | R$ 123,45 | gasto"`.
- `embedar(textos) -> np.ndarray` via `ollama.embed` com `nomic-embed-text`, normalizado.
- `buscar(pergunta, indice, k=10) -> lista de documentos` por similaridade de cosseno.
- `responder(pergunta, ...)`: monta o prompt com os documentos recuperados e pede para o modelo
  responder **só** com base neles.
- Cache dos embeddings em `data/embeddings.npy`, porque refazer 1.050 embeddings a cada execução é lento.
- No notebook: perguntas de busca que funcionam e uma de soma que **falha**. A falha é o aprendizado.
- Testes: `buscar` com vetores falsos (sem Ollama) retorna os mais próximos na ordem certa.

### 04 — Tools
- Expor funções pandas como ferramentas (`total_gasto(inicio, fim)`, `gastos_por_mes()`,
  `buscar_estabelecimento(nome)`) via tool calling do `ollama.chat`.
- O modelo decide a ferramenta; o Python executa e devolve o resultado; o modelo redige a resposta.
- Comparar lado a lado com a resposta do RAG da etapa 03 para a mesma pergunta de agregação.

## Tratamento de erros

- Ollama fora do ar: capturar o erro de conexão e mostrar a mensagem "abra o app Ollama".
- Modelo não baixado: mostrar o comando `ollama pull <modelo>`.
- Sem tratamento além disso, porque é um projeto de estudo.

## Fora de escopo

- Interface web/chat app, banco vetorial, categorização automática por LLM, frameworks de RAG,
  avaliação formal de qualidade. Podem virar extensões depois.

## Dependências novas

- `ollama` (runtime), `pytest` (dev).

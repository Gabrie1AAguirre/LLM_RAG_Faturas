# RAG local sobre faturas — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pacote `faturas` + 4 notebooks didáticos que levam do chat puro com LLM local ao RAG e ao tool calling sobre as faturas Nubank.

**Architecture:** Lógica reutilizável em `src/faturas/` (dados, llm, rag, tools), testada com pytest sem precisar do Ollama (o cliente é mockado). Os notebooks importam o pacote e explicam cada etapa em markdown em português.

**Tech Stack:** Python 3.14, uv, pandas, numpy, cliente `ollama` (Python), Ollama local com `qwen3:4b` e `nomic-embed-text`, pytest.

**Spec:** `docs/superpowers/specs/2026-10-17-rag-faturas-design.md`

## Global Constraints

- Tudo local: só `http://localhost:11434`; nenhuma chamada de rede para outros hosts.
- Modelo de chat padrão `qwen3:4b`; embeddings `nomic-embed-text`. São constantes `MODELO_CHAT` e `MODELO_EMBED` em `src/faturas/llm.py`.
- Chamadas de chat usam `think=False`; se vier `<think>...</think>` no texto, remover antes de exibir.
- `data/` nunca é commitado (já está no `.gitignore`). Isso vale também para `data/embeddings.npy`.
- Notebooks são commitados **sem saídas** (as saídas mostram transações reais). Verificação executa para `$TEMP/nbcheck`, nunca `--inplace`; antes do `git add`, rodar `uv run jupyter nbconvert --clear-output --inplace notebooks/<nome>.ipynb`.
- Python 3.14 / `uv`. Comandos rodam com `uv run ...`.
- Textos para o usuário (mensagens de erro, markdown dos notebooks) em português.
- Desvio do spec: um módulo extra `src/faturas/llm.py` concentra a conexão com o Ollama e o tratamento de erros, usado pelas etapas 01, 03 e 04.

## Review Focus

1. **Compras idênticas legítimas no mesmo dia** (dois cafés iguais): a deduplicação entre arquivos não pode apagá-las. Teste em Task 2.
2. **CSV novo adicionado depois do cache de embeddings**: o cache precisa ser recalculado, e não pode reaproveitar vetores velhos. Teste em Task 4.
3. **Ollama fechado ou modelo não baixado**: a mensagem deve ser clara ("abra o app Ollama" / `ollama pull <modelo>`), sem traceback de httpx. Teste em Task 3.
4. **Valor com `R$`, `\xa0` ou espaço após o sinal** (`"- 2.345,67"`): a conversão deve ser correta. Teste em Task 1.
5. **Tool chamada pelo modelo com nome em caixa/acentuação diferente** ("pizzaria exemplo" vs "Pizzaria Exemplo"): a busca deve encontrar. Teste em Task 5.

---

### Task 1: Setup do pacote + `parse_amount`

**Files:**
- Modify: `pyproject.toml`
- Create: `src/faturas/__init__.py` (vazio), `src/faturas/dados.py`
- Test: `tests/test_dados.py`

**Interfaces:**
- Produces: `parse_amount(x: str | float | int) -> float`

- [ ] **Step 1: Configurar o pyproject**

Adicionar `[build-system]` com hatchling (`requires = ["hatchling"]`, `build-backend = "hatchling.build"`) e `[tool.hatch.build.targets.wheel] packages = ["src/faturas"]`. Rodar `uv add ollama` e `uv add --dev pytest`. Adicionar `[tool.pytest.ini_options] testpaths = ["tests"]`.

Run: `uv sync && uv run python -c "import faturas, ollama; print('ok')"`
Expected: `ok`

- [ ] **Step 2: Escrever os testes que falham**

```python
import pytest
from faturas.dados import parse_amount

@pytest.mark.parametrize("entrada, esperado", [
    ("123,45", 123.45),
    ("- 2.345,67", -2345.67),
    ("1.234,56", 1234.56),
    ("R$\xa067,89", 45.72),
    (31.5, 31.5),
    (10, 10.0),
    ("12.50", 12.5),        # CSV que já vem com ponto decimal
])
def test_parse_amount(entrada, esperado):
    assert parse_amount(entrada) == pytest.approx(esperado)
```

- [ ] **Step 3: Rodar e ver falhar**

Run: `uv run pytest tests/test_dados.py -v`
Expected: FAIL (`ImportError` / `ModuleNotFoundError`)

- [ ] **Step 4: Implementar `parse_amount` em `src/faturas/dados.py`**

Copiar a lógica do `notebooks/EDA.ipynb`: número passa direto como float; senão manter só dígitos, `,`, `.` e `-`; se houver vírgula, o ponto é separador de milhar e a vírgula é o decimal.

- [ ] **Step 5: Rodar e ver passar**

Run: `uv run pytest tests/test_dados.py -v`
Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/ tests/
git commit -m "feat: pacote faturas e parse_amount"
```

---

### Task 2: `carregar_faturas` + notebook 02

**Files:**
- Modify: `src/faturas/dados.py`
- Test: `tests/test_dados.py`
- Create: `notebooks/02_dados.ipynb`

**Interfaces:**
- Consumes: `parse_amount`
- Produces: `carregar_faturas(pasta: str | Path) -> pd.DataFrame` com colunas exatas `date` (datetime64), `title` (str), `amount` (float), `fatura` (datetime64, extraída de `Nubank_AAAA-MM-DD.csv`), `tipo` (`"gasto"` se `amount > 0`, senão `"credito"`), ordenado por `date` com índice resetado.

- [ ] **Step 1: Escrever os testes que falham**

Uma fixture escreve CSVs em `tmp_path` (cabeçalho `date,title,amount`).

```python
def test_carregar_junta_e_tipa(tmp_path):
    # Nubank_2026-09-08.csv: 2026-09-01,Padaria,"10,00"  e  2026-09-02,Pagamento recebido,"- 3.456,00"
    # Nubank_2026-10-12.csv: 2026-10-04,Pizzaria Exemplo,"123,45"
    df = carregar_faturas(tmp_path)
    assert list(df.columns) == ["date", "title", "amount", "fatura", "tipo"]
    assert len(df) == 3
    assert df["tipo"].tolist() == ["gasto", "credito", "gasto"]
    assert df.loc[2, "fatura"] == pd.Timestamp("2026-10-12")

def test_remove_duplicata_entre_arquivos(tmp_path):
    # a mesma linha 2026-09-30,Pet Shop Exemplo,"67,89" em dois arquivos diferentes
    assert len(carregar_faturas(tmp_path)) == 1

def test_mantem_compras_iguais_no_mesmo_arquivo(tmp_path):
    # duas linhas idênticas 2026-10-01,Cafe,"5,00" no MESMO arquivo
    assert len(carregar_faturas(tmp_path)) == 2

def test_arquivo_so_com_cabecalho(tmp_path):
    # um arquivo só com o cabeçalho + um arquivo normal com 1 linha
    assert len(carregar_faturas(tmp_path)) == 1
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_dados.py -v`
Expected: os 4 testes novos FAIL (`ImportError: carregar_faturas`)

- [ ] **Step 3: Implementar `carregar_faturas`**

Deduplicação que preserva repetições dentro de um mesmo arquivo:
```python
# por arquivo: n = ocorrência da linha (date,title,amount) dentro daquele arquivo
df["n"] = df.groupby(["fatura", "date", "title", "amount"]).cumcount()
df = df.drop_duplicates(subset=["date", "title", "amount", "n"]).drop(columns="n")
```
Ler os arquivos com `dtype={"amount": str}` antes de aplicar `parse_amount`, e ordenar os arquivos pelo nome.

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest -v`
Expected: 11 passed

- [ ] **Step 5: Criar `notebooks/02_dados.ipynb`**

Células: markdown explicando por que os dados precisam ser limpos antes de ir para um LLM; `carregar_faturas("../data")`; `df.info()`; quantas linhas e quantas duplicatas foram removidas (comparar com o total bruto de linhas dos CSVs); `df.groupby("tipo")["amount"].agg(["count", "sum"])`; um exemplo de linha formatada como texto, antecipando a etapa 03.

Run: `uv run jupyter nbconvert --to notebook --execute --output-dir "$TEMP/nbcheck" notebooks/02_dados.ipynb`
Expected: termina sem erro

- [ ] **Step 6: Commit**

```bash
git add src/faturas/dados.py tests/test_dados.py notebooks/02_dados.ipynb
git commit -m "feat: carregar_faturas com dedupe entre arquivos"
```

---

### Task 3: `llm.py` + notebook 01 (chat sem RAG)

**Files:**
- Create: `src/faturas/llm.py`, `notebooks/01_chat_sem_rag.ipynb`
- Test: `tests/test_llm.py`

**Interfaces:**
- Produces:
  - `MODELO_CHAT = "qwen3:4b"`, `MODELO_EMBED = "nomic-embed-text"`
  - `class OllamaIndisponivel(RuntimeError)`, com a mensagem pronta para o usuário
  - `remover_think(texto: str) -> str`
  - `conversar(mensagens: list[dict], modelo: str = MODELO_CHAT, **kwargs) -> str`: chama `ollama.chat(model=modelo, messages=mensagens, think=False, **kwargs)` e devolve `remover_think(resposta.message.content)`. Os `kwargs` passam `tools` etc. adiante.
  - `chamar_ollama(fn, *args, modelo: str, **kwargs)`: executa `fn(*args, **kwargs)` traduzindo os erros. `ConnectionError` vira `OllamaIndisponivel("Não consegui falar com o Ollama. Abra o app Ollama e tente de novo.")`. `ollama.ResponseError` com status 404 vira `OllamaIndisponivel(f"Modelo '{modelo}' não encontrado. Rode: ollama pull {modelo}")`. `conversar` e `embedar` (Task 4) usam essa função.

- [ ] **Step 1: Escrever os testes que falham**

```python
def test_remover_think():
    assert remover_think("<think>\nblá\n</think>\n\nok") == "ok"
    assert remover_think("sem think") == "sem think"

def test_ollama_fechado(monkeypatch):
    def falha(**kw): raise ConnectionError("refused")
    monkeypatch.setattr(ollama, "chat", falha)
    with pytest.raises(OllamaIndisponivel, match="Abra o app Ollama"):
        conversar([{"role": "user", "content": "oi"}])

def test_modelo_nao_baixado(monkeypatch):
    def falha(**kw): raise ollama.ResponseError("model not found", 404)
    monkeypatch.setattr(ollama, "chat", falha)
    with pytest.raises(OllamaIndisponivel, match="ollama pull qwen3:4b"):
        conversar([{"role": "user", "content": "oi"}])
```

Observação: `llm.py` deve chamar `ollama.chat` pelo módulo (`import ollama`), e não via `from ollama import chat`, para o monkeypatch funcionar.

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_llm.py -v`
Expected: FAIL (`ModuleNotFoundError: faturas.llm`)

- [ ] **Step 3: Implementar `src/faturas/llm.py`** conforme as Interfaces.

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest -v`
Expected: 14 passed

- [ ] **Step 5: Criar `notebooks/01_chat_sem_rag.ipynb`**

Células, com markdown curto antes de cada uma:
1. O que é um LLM e o que é o Ollama (servidor local, `localhost:11434`).
2. `conversar` com "Explique RAG em 2 frases".
3. Streaming token a token com `ollama.chat(..., stream=True, think=False)` direto, para mostrar o que acontece por baixo.
4. "Quanto gastei no Pizzaria Exemplo em outubro de 2026?" **sem dados**: o modelo inventa ou se recusa. Markdown: alucinação.
5. Colar no prompt as 20 transações mais recentes (via `carregar_faturas`) e repetir: agora acerta.
6. Markdown + código: estimar tokens de todas as ~1.050 linhas (`len(texto) / 4` como aproximação) vs a janela de 4096 tokens. Conclusão: não cabe, e é por isso que existe o RAG.

Run: `uv run jupyter nbconvert --to notebook --execute --output-dir "$TEMP/nbcheck" notebooks/01_chat_sem_rag.ipynb` (com o Ollama aberto)
Expected: termina sem erro

- [ ] **Step 6: Commit**

```bash
git add src/faturas/llm.py tests/test_llm.py notebooks/01_chat_sem_rag.ipynb
git commit -m "feat: cliente llm com erros amigáveis e notebook 01"
```

---

### Task 4: `rag.py` + notebook 03

**Files:**
- Create: `src/faturas/rag.py`, `notebooks/03_rag.ipynb`
- Test: `tests/test_rag.py`

**Interfaces:**
- Consumes: `carregar_faturas`, `MODELO_EMBED`, `MODELO_CHAT`, `conversar`, `chamar_ollama`
- Produces:
  - `transacao_para_texto(linha: pd.Series) -> str`, no formato exato `"2026-10-04 | Pizzaria Exemplo | R$ 123,45 | gasto"` (valor em formato pt-BR com milhar, ex.: `R$ -2.345,67`)
  - `embedar(textos: list[str], prefixo: str) -> np.ndarray`: `ollama.embed(model=MODELO_EMBED, input=[prefixo + t for t in textos])`, devolve vetores normalizados (norma 1), shape `(n, d)`. Os prefixos do nomic são `"search_document: "` para documentos e `"search_query: "` para perguntas.
  - `@dataclass Indice: textos: list[str]; vetores: np.ndarray`
  - `carregar_ou_criar_indice(df: pd.DataFrame, cache: Path) -> Indice`: o cache é um `.npz` com `textos` e `vetores`. Reusa o cache **só se** os textos salvos forem iguais aos atuais; senão recalcula e regrava.
  - `buscar(pergunta: str, indice: Indice, k: int = 10) -> list[str]`: similaridade = `indice.vetores @ embedar([pergunta], "search_query: ")[0]`, devolve os top-k textos em ordem decrescente.
  - `responder(pergunta: str, indice: Indice, k: int = 10) -> str`: system prompt manda responder **só** com as transações fornecidas e dizer "não sei" se não estiver lá; a mensagem do usuário contém as transações recuperadas + a pergunta; usa `conversar`.

- [ ] **Step 1: Escrever os testes que falham**

Mock de `ollama.embed` com vetores determinísticos: um dicionário texto → vetor; o objeto retornado tem o atributo `.embeddings`.

```python
def test_transacao_para_texto():
    linha = pd.Series({"date": pd.Timestamp("2026-10-04"), "title": "Pizzaria Exemplo",
                       "amount": 123.45, "tipo": "gasto"})
    assert transacao_para_texto(linha) == "2026-10-04 | Pizzaria Exemplo | R$ 123,45 | gasto"
    linha2 = linha.copy(); linha2["amount"] = -2345.67; linha2["tipo"] = "credito"
    assert transacao_para_texto(linha2).endswith("R$ -2.345,67 | credito")

def test_buscar_ordena_por_similaridade(monkeypatch):
    # índice com vetores [1,0], [0,1], [0.7,0.7]; pergunta embeda para [1,0]
    assert buscar("q", indice, k=2) == ["A", "C"]

def test_embedar_normaliza(monkeypatch):
    # mock devolve [[3, 4]] -> resultado [[0.6, 0.8]]
    np.testing.assert_allclose(embedar(["x"], "search_document: "), [[0.6, 0.8]])

def test_cache_invalida_quando_dados_mudam(tmp_path, monkeypatch):
    # 1ª chamada com df de 2 linhas cria cache; contador de chamadas ao embed == 1
    # 2ª chamada com o mesmo df: contador continua 1 (usou cache)
    # 3ª chamada com df de 3 linhas: contador == 2 e len(indice.textos) == 3
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_rag.py -v`
Expected: FAIL (`ModuleNotFoundError: faturas.rag`)

- [ ] **Step 3: Implementar `src/faturas/rag.py`** conforme as Interfaces.

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest -v`
Expected: 18 passed

- [ ] **Step 5: Criar `notebooks/03_rag.ipynb`** (rodar `ollama pull nomic-embed-text` antes)

Células com markdown explicando cada conceito:
1. O que é embedding: embedar "padaria", "pão" e "gasolina" e mostrar a matriz de similaridade 3x3.
2. Transações viram texto (`transacao_para_texto`, 5 exemplos).
3. `carregar_ou_criar_indice(df, Path("../data/embeddings.npz"))`, com o tempo da 1ª vez vs da 2ª.
4. `buscar("restaurante", indice)`: mostrar o que voltou e por quê.
5. `responder` para 3 perguntas de **busca** que funcionam (ex.: "Quando comprei no Pizzaria Exemplo?", "Quais gastos com pet?", "Tenho alguma assinatura?").
6. `responder("Quanto gastei no total em 2025?")` → resposta errada. Comparar com o valor real do pandas (`df[(df.tipo=="gasto") & (df.date.dt.year==2025)].amount.sum()`). Markdown: o RAG só vê k=10 linhas e o LLM não é calculadora. Gancho para a etapa 04.

Run: `uv run jupyter nbconvert --to notebook --execute --output-dir "$TEMP/nbcheck" notebooks/03_rag.ipynb`
Expected: termina sem erro; `data/embeddings.npz` criado e ignorado pelo git (`git status` não o mostra)

- [ ] **Step 6: Commit**

```bash
git add src/faturas/rag.py tests/test_rag.py notebooks/03_rag.ipynb
git commit -m "feat: RAG com embeddings em numpy e cache, notebook 03"
```

---

### Task 5: `tools.py` + notebook 04

**Files:**
- Create: `src/faturas/tools.py`, `notebooks/04_tools.ipynb`
- Test: `tests/test_tools.py`

**Interfaces:**
- Consumes: `conversar`, `MODELO_CHAT`, DataFrame de `carregar_faturas`
- Produces: `criar_ferramentas(df: pd.DataFrame) -> list[Callable]` devolve 3 funções fechadas sobre `df`, cada uma com type hints e docstring em estilo Google (o cliente `ollama` gera o schema da tool a partir delas):
  - `total_gasto(inicio: str, fim: str) -> float`: soma de `amount` com `tipo == "gasto"` e `inicio <= date <= fim` (datas `"AAAA-MM-DD"`, inclusivas), arredondada para 2 casas
  - `gastos_por_mes() -> dict[str, float]`: `{"2025-01": 1234.56, ...}`, só gastos
  - `buscar_estabelecimento(nome: str) -> list[dict]`: linhas cujo `title` contém `nome`, ignorando caixa e acentos (normalizar com `unicodedata` NFKD + remover combining), cada uma como `{"date": "AAAA-MM-DD", "title": ..., "amount": ...}`
- Produces: `perguntar_com_tools(pergunta: str, df: pd.DataFrame, max_rodadas: int = 5) -> str`: loop em que `ollama.chat` recebe `tools=ferramentas`; se a resposta tiver `message.tool_calls`, executa cada uma pelo nome com `function.arguments`, anexa `{"role": "tool", "tool_name": nome, "content": json.dumps(resultado, ensure_ascii=False)}` e repete; sem tool_calls, devolve `remover_think(content)`. Nome de tool desconhecido → anexa `{"erro": "ferramenta X não existe"}` em vez de quebrar. O system prompt informa a data de hoje e diz para usar as ferramentas para qualquer número.

- [ ] **Step 1: Escrever os testes que falham**

DataFrame fixo em fixture: Pizzaria Exemplo 123,45 em 2025-03-10; Padaria 10,00 em 2025-12-31; Pagamento -3.456,00 em 2025-06-07; Padaria 20,00 em 2026-01-02.

```python
def test_total_gasto_inclusivo_e_ignora_credito(ferramentas):
    assert ferramentas["total_gasto"]("2025-01-01", "2025-12-31") == 133.45

def test_gastos_por_mes(ferramentas):
    assert ferramentas["gastos_por_mes"]() == {"2025-03": 123.45, "2025-12": 10.0, "2026-01": 20.0}

def test_buscar_ignora_caixa_e_acento(ferramentas):
    assert len(ferramentas["buscar_estabelecimento"]("pizzaria exemplo")) == 1
    assert len(ferramentas["buscar_estabelecimento"]("PADARÍA")) == 2

def test_loop_executa_tool_e_responde(monkeypatch, df):
    # mock de ollama.chat: 1ª chamada devolve tool_call total_gasto(2025-01-01, 2025-12-31);
    # 2ª chamada verifica que a última mensagem é role "tool" com content "133.45" e devolve "R$ 133,45"
    assert perguntar_com_tools("quanto gastei em 2025?", df) == "R$ 133,45"

def test_tool_desconhecida_nao_quebra(monkeypatch, df):
    # 1ª chamada pede tool "inventada"; 2ª vê content com "não existe" e devolve "desculpe"
    assert perguntar_com_tools("x", df) == "desculpe"
```

`ferramentas` é a fixture `{f.__name__: f for f in criar_ferramentas(df)}`.

- [ ] **Step 2: Rodar e ver falhar**

Run: `uv run pytest tests/test_tools.py -v`
Expected: FAIL (`ModuleNotFoundError: faturas.tools`)

- [ ] **Step 3: Implementar `src/faturas/tools.py`** conforme as Interfaces. A chamada a `ollama.chat` passa por `chamar_ollama` para manter os erros amigáveis.

- [ ] **Step 4: Rodar e ver passar**

Run: `uv run pytest -v`
Expected: 23 passed

- [ ] **Step 5: Criar `notebooks/04_tools.ipynb`**

1. Markdown: o que é tool calling (o modelo escolhe a função; quem executa é o Python).
2. Mostrar o schema JSON que o cliente gera a partir de `total_gasto` (o modelo só vê isso).
3. Executar o loop passo a passo uma vez (imprimindo cada `tool_call` e cada resultado), sem usar `perguntar_com_tools`, para o mecanismo ficar visível.
4. `perguntar_com_tools` para "Quanto gastei em 2025?", "Qual mês gastei mais?" e "Quanto gastei no Pizzaria Exemplo?".
5. Tabela comparando RAG (notebook 03) vs tools vs valor real do pandas para "Quanto gastei em 2025?". Markdown de conclusão: quando usar cada abordagem.

Run: `uv run jupyter nbconvert --to notebook --execute --output-dir "$TEMP/nbcheck" notebooks/04_tools.ipynb`
Expected: termina sem erro

- [ ] **Step 6: Commit**

```bash
git add src/faturas/tools.py tests/test_tools.py notebooks/04_tools.ipynb
git commit -m "feat: tool calling com pandas, notebook 04"
```

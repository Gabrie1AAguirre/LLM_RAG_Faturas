"""As saídas dos notebooks mostram transações reais: elas nunca podem chegar ao git."""

import json
import subprocess
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((RAIZ / "notebooks").glob("*.ipynb"))


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=RAIZ, capture_output=True, text=True, check=True,
                          encoding="utf-8").stdout


def como_o_git_guardaria(caminho: Path) -> dict:
    # hash-object --path aplica os filtros de .gitattributes, exatamente como um `git add`
    relativo = caminho.relative_to(RAIZ).as_posix()
    blob = git("hash-object", "-w", f"--path={relativo}", relativo).strip()
    return json.loads(git("cat-file", "blob", blob))


@pytest.mark.parametrize("caminho", NOTEBOOKS, ids=lambda p: p.name)
def test_git_nunca_guarda_saidas_de_notebook(caminho):
    celulas = como_o_git_guardaria(caminho)["cells"]
    com_saida = [i for i, c in enumerate(celulas) if c.get("outputs")]
    assert not com_saida, (
        f"{caminho.name}: as células {com_saida} iriam para o git com saídas (dados reais). "
        "Rode: uv run nbstripout --install --attributes .gitattributes"
    )


def test_dados_ignorados():
    assert git("check-ignore", "data/qualquer.csv", "data/embeddings.npz").split() == [
        "data/qualquer.csv", "data/embeddings.npz",
    ]

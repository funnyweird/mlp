# MLP do Zero — MNIST

## Como Rodar

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train.py
```

## Arquitetura Escolhida

- **Camadas ocultas:** 2 (784 → 256 → 128 → 10)
- **Ativação:** ReLU nas camadas ocultas, Softmax na saída
- **Loss:** Cross-Entropy
- **Otimizadores:** SGD e Adam

> *Por que essas escolhas?*

## Resultados

| Configuração | Acurácia (teste) | Épocas |
|---|---|---|
| SGD, lr=0.01 | — | — |
| Adam, lr=0.001 | — | — |

> *Curva de loss* 

## Decisões e Dificuldades

Durante o desenvolvimento, o VSCode reiniciou inesperadamente e desligou o autosave sem eu perceber. Fiz vários commits com arquivos vazios antes de notar que nada tinha sido salvo em disco. Aprendi a sempre verificar com `git diff` antes de commitar e a manter o autosave ativo.
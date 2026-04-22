# 🎯 Churn Prediction — POC

Modelo de Machine Learning para identificar clientes com alta probabilidade de cancelar a assinatura, permitindo que o time de Customer Success atue preventivamente.

## O Problema

Em SaaS, perder clientes (churn) significa perder receita recorrente. Adquirir um novo cliente custa 5-7x mais do que reter um existente. Esse projeto usa dados comportamentais e contratuais dos clientes para prever quem está em risco de cancelamento.

## Dataset

Dataset sintético com **3.000 clientes** e 13 features simulando uma empresa SaaS:

| Feature | Descrição |
|---|---|
| `tenure_meses` | Tempo de casa (meses) |
| `plano` | Basic / Pro / Enterprise |
| `mensalidade` | Valor pago por mês (R$) |
| `contrato` | Mensal / Anual / Bienal |
| `usuarios_ativos` | Usuários ativos na conta |
| `logins_30d` | Logins nos últimos 30 dias |
| `features_usadas` | Quantas features do produto usa |
| `tickets_suporte` | Tickets abertos (últimos 90 dias) |
| `nps` | Net Promoter Score (0-10) |
| `atraso_pagamento_dias` | Média de atraso no pagamento |
| `desconto_ativo` | Se possui desconto promocional |
| `integracoes_ativas` | Nº de integrações com ferramentas externas |
| **`churn`** | **Target: 1 = cancelou, 0 = ativo** |

Taxa de churn: **18.3%** (dados desbalanceados).

## Pipeline

```
1. Análise Exploratória (EDA)
   └─ Gráficos de churn por contrato, plano, desconto
   └─ Comparação de médias (groupby churn)
   └─ Matriz de correlação (heatmap)

2. Pré-processamento
   └─ Remoção de colunas sem poder preditivo (cliente_id)
   └─ One-Hot Encoding (plano, contrato)
   └─ Separação treino/teste (75/25, estratificado)

3. Feature Scaling
   └─ StandardScaler (fit no treino, transform no teste)

4. PCA
   └─ 10 componentes = 91.3% da variância explicada

5. Modelagem (4 algoritmos comparados)
   └─ Regressão Logística
   └─ Árvore de Decisão
   └─ Random Forest
   └─ KNN

6. Avaliação e interpretação
```

## Resultados

| Modelo | Acurácia | Precision (Churn) | Recall (Churn) | F1 (Churn) |
|---|---|---|---|---|
| **Regressão Logística** | 71% | 36% | **80%** | 0.49 |
| Árvore de Decisão | 69% | 32% | 61% | 0.42 |
| Random Forest | 82% | 55% | 9% | 0.15 |
| KNN | 81% | 42% | 7% | 0.12 |

### Aprendizado-chave

Random Forest e KNN tinham a maior acurácia (82% e 81%), mas detectavam **menos de 10%** dos churns reais. Eles simplesmente chutavam "não churn" para quase todo mundo, a acurácia alta era uma ilusão causada pelo desbalanceamento dos dados.

A **Regressão Logística** com `class_weight='balanced'` foi o melhor modelo: recall de **80%** (detectou 4 de cada 5 churns reais). Em churn, priorizar recall é mais importante porque é pior perder um cliente sem saber do que contatar um que ia ficar.

## Principais fatores de churn

Fatores que **aumentam** churn:
- Contrato mensal (26% de churn vs 3.5% no bienal)
- Muitos tickets de suporte
- Desconto ativo (cliente fica pelo preço, não pelo valor)

Fatores que **protegem** contra churn:
- NPS alto
- Mais logins (engajamento)
- Uso de mais features do produto
- Contrato bienal
- Integrações ativas (lock-in)

## Como rodar

```bash
# Clone o repositório
git clone https://github.com/ViniciusGeisler/churn-prediction-poc.git

# Abra o notebook
# Opção 1: Google Colab (recomendado)
# Suba o notebook e o CSV no Colab

# Opção 2: Local
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook Modelo_ML_para_churn.ipynb
```

## Estrutura do projeto

```
├── README.md
├── churn_saas.csv                  # Dataset sintético
├── Modelo_ML_para_churn.ipynb      # Notebook com o pipeline completo
```

## Tecnologias

- Python 3
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (LogisticRegression, DecisionTree, RandomForest, KNN, PCA, StandardScaler)

---

*Projeto desenvolvido como estudo prático durante pós-graduação em Inteligência Artificial.*

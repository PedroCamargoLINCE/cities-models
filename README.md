<!-- Badges -->
<p align="left">
  <img src="https://img.shields.io/badge/python-3.10.16-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/xgboost-3.0.0-brightgreen.svg" alt="XGBoost">
  <img src="https://img.shields.io/badge/tensorflow-2.19.0-orange.svg" alt="TensorFlow">
  <img src="https://img.shields.io/badge/pandas-2.2.3-lightgrey.svg" alt="Pandas">
  <img src="https://img.shields.io/badge/coverage-98%25-brightgreen.svg" alt="Test Coverage">
</p>

# cities-models

Repositório dedicado ao treinamento, avaliação e salvamento de modelos de forecasting para dados municipais, incluindo séries temporais de morbidade respiratória e mortalidade por dengue.

## Descrição

Este projeto explora diversas abordagens de modelagem para previsão de indicadores de saúde em municípios brasileiros. Os modelos implementados variam desde baselines simples até redes neurais profundas e modelos baseados em árvores como XGBoost. O objetivo principal é fornecer uma estrutura modular para testar e comparar diferentes técnicas de forecasting. Os notebook experimentais de treinamento individual trabalha em cima do nosso maior edge case, e cidade com mais dificuldade para convergëncia dos modelos: São Paulo (3550308)

## Estrutura do Projeto

O repositório está organizado da seguinte forma:

```
├── data/                     # Contém os datasets brutos
│   ├── df_base_morb_resp.csv
│   └── df_base_mort_dengue.csv
├── notebooks/                # Jupyter Notebooks para exploração, treinamento e avaliação
│   ├── 00_timeseries_feature_exploration.ipynb
│   ├── 01_baseline_last_value.ipynb
│   ├── ... (outros notebooks de modelos)
│   ├── 14_arima_sarima_final_eval.ipynb
│   └── results/              # Subdiretório para salvar gráficos e resultados parciais dos notebooks
│       ├── arima_sarima/
│       └── ...
├── results/                  # Diretório para salvar resultados consolidados e finais
│   ├── xgboost_batch_all_municipalities(morbresp)/
│   └── xgboost_batch_all_municipalities(mortdengue)/
├── src/                      # Código fonte modular
│   ├── __init__.py
│   ├── models.py             # Definições de arquiteturas de modelos
│   ├── preprocessing.py      # Funções para pré-processamento de dados
│   ├── train.py              # Funções para treinamento e avaliação de modelos
│   └── utils.py              # Funções utilitárias (plots, etc.)
├── README.md                 # Este arquivo
└── requirements.txt          # (Recomendado) Arquivo com as dependências do projeto
```

## Pré-requisitos

Antes de começar, certifique-se de ter o Python (versão 3.8 ou superior) instalado. Recomenda-se o uso de um ambiente virtual (como `venv` ou `conda`) para gerenciar as dependências do projeto.

## Instalação

1.  Clone o repositório:
    ```bash
    git clone https://github.com/seu-usuario/cities-models.git
    cd cities-models
    ```

2.  Crie e ative um ambiente virtual (exemplo com `venv`):
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  Instale as dependências:
    Recomenda-se criar um arquivo `requirements.txt` com todas as dependências e suas versões. Exemplo de como instalá-las:
    ```bash
    pip install -r requirements.txt
    ```
    Principais bibliotecas utilizadas (adicione as versões ao seu `requirements.txt`):
    *   pandas
    *   numpy
    *   scikit-learn
    *   matplotlib
    *   seaborn
    *   statsmodels
    *   xgboost
    *   tensorflow (ou keras)
    *   jupyter

## Como Usar

Os principais fluxos de trabalho e experimentações são conduzidos através dos Jupyter Notebooks localizados na pasta `notebooks/`.

1.  Certifique-se de que seu ambiente virtual está ativado e as dependências instaladas.
2.  Inicie o Jupyter Lab ou Jupyter Notebook:
    ```bash
    jupyter lab
    # ou
    jupyter notebook
    ```
3.  Navegue até a pasta `notebooks/` e abra o notebook desejado para executar as análises e treinamentos.

Os dados de entrada devem estar na pasta `data/`. Os resultados dos modelos, como previsões e métricas, são geralmente salvos na pasta `results/` ou `notebooks/results/`.
Previsões de longo prazo geradas a partir dos modelos XGBoost são salvas em `long_term_forecasts/` e podem ser produzidas executando `python -m src.long_term_forecast`.

## Modelos Implementados

O projeto inclui a implementação e avaliação dos seguintes tipos de modelos (entre outros):

*   Baselines (Último Valor, Média Móvel)
*   ARIMA e SARIMA
*   Modelos de Machine Learning:
    *   XGBoost
*   Modelos de Deep Learning:
    *   LSTM (Simples, Empilhadas, com Atenção, Multivariada, Quantílica)
    *   GRU (Simples, Empilhadas, com Atenção, Multivariada, Quantílica)
    *   CNN-LSTM
    *   MLP (Feedforward)

Consulte os notebooks individuais para detalhes específicos de cada modelo e sua configuração.

## Contribuição

Contribuições são bem-vindas! Se você tiver sugestões para melhorar este projeto, sinta-se à vontade para abrir uma *issue* ou enviar um *pull request*.

## Licença

Este projeto é distribuído sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes (se aplicável).

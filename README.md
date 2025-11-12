# Analisador de Filmes

## APLICAÇÃO WEB PARA ANÁLISE DE DADOS E MACHINE LEARNING

Este projeto é uma aplicação web completa desenvolvida em Python e Django, criada como trabalho final de disciplina. A plataforma permite que qualquer usuário faça o upload de um arquivo `.csv`, receba uma análise exploratória de dados automática e interativa, e ainda treine e teste modelos de Machine Learning para classificação.

---

## Funcionalidades Principais

A aplicação é dividida em três etapas principais:

### 1. Upload de CSV
* Interface moderna de "arrastar e soltar" para upload de arquivos `.csv`.
* Os dados são processados com Pandas e armazenados de forma eficiente na sessão do usuário para uso nas próximas etapas.

### 2. Análise Exploratória Automática
Assim que o upload é feito, o usuário é direcionado para uma página de análise que gera automaticamente um relatório visual completo dos dados. A classe `DataAnalyzer` identifica os tipos de colunas (numéricas, categóricas, datas, geográficas) e gera:

* Histogramas, boxplots e tabelas de estatísticas descritivas (média, mediana, desvio padrão, etc.).
* Gráficos de barras com as categorias mais frequentes e gráficos de pizza para colunas com poucas categorias únicas.
* Heatmaps de correlação (para todas as colunas numéricas) e gráficos de dispersão (scatter plots) para os pares mais correlacionados.
* Gráficos de linha para séries temporais (contagem de eventos por dia) e gráficos de média móvel para identificar tendências.
* Mapas de dispersão interativos (se colunas `lat`/`lon` forem detectadas) ou gráficos de barras para colunas de localização (como país, cidade, etc.).

### 3. Predição com Machine Learning
A página de predição permite ao usuário construir, treinar e testar modelos de classificação usando os dados do CSV (onde a última coluna é tratada como o "alvo" ou *target*):

* **Modelos Suportados:**
    * K-Nearest Neighbors (KNN)
    * Decision Tree (Árvore de Decisão)
    * Random Forest
    * Logistic Regression (Regressão Logística)
    * SVM (Support Vector Machine)
* **Pipeline de Pré-processamento:** Um pipeline robusto do `scikit-learn` é aplicado automaticamente, tratando dados nulos (`SimpleImputer`), padronizando dados numéricos (`StandardScaler`) e convertendo colunas categóricas em *dummies* (`OneHotEncoder`).
* **Ações do Usuário:**
    1.  **Treinar:** O modelo é treinado em 80% dos dados e avaliado em 20% (split 80/20).
    2.  **Ajustar Hiperparâmetros:** A interface permite que o usuário insira valores para os hiperparâmetros de cada modelo (ex: `n_neighbors` no KNN ou `max_depth` na Árvore).
    3.  **Prever:** O usuário pode preencher um formulário com novos dados para obter uma predição em tempo real do modelo treinado.

---

## Tecnologias Utilizadas

Este projeto integra bibliotecas de ponta do ecossistema Python para web e ciência de dados:

| Categoria | Tecnologia | Propósito |
| :--- | :--- | :--- |
| **Backend** | Django | Framework web principal para estruturar o projeto, gerenciar URLs e views. |
| **Análise de Dados**| Pandas | Leitura do CSV, limpeza, manipulação e análise de dados. |
| | NumPy | Suporte para operações numéricas e utilizado pelo Pandas/Scikit-learn. |
| **Visualização** | Plotly | Geração de gráficos interativos (heatmaps, mapas, scatter plots). |
| | Matplotlib | Geração de gráficos estáticos (usado como *fallback* ou para análises rápidas). |
| **Machine Learning**| Scikit-learn | Implementação de todos os modelos de ML e pipelines de pré-processamento. |
| **Utilitários** | Scipy | Dependência do Scikit-learn para operações científicas. |

---

## Como Rodar o Projeto

O Python e o `pip` devem estar instalados.

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/realcloso/analisador-de-filmes.git
    cd analisador-de-filmes
    ```

2.  **Crie e ative um ambiente virtual (venv):**
    *No Windows:*
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
    *No macOS/Linux:*
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```
   

4.  **Entre na pasta do projeto Django:**
    (A pasta que contém o `manage.py`)
    ```bash
    cd trabalhofinal
    ```

5.  **Aplique as migrações do banco de dados:**
    (necessário para o Django funcionar)
    ```bash
    python manage.py migrate
    ```

6.  **Inicie o servidor de desenvolvimento:**
    ```bash
    python manage.py runserver
    ```
   

7.  Abra seu navegador e acesse: **`http://127.0.0.1:8000/`**

---
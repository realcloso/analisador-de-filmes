# analisador-de-filmes

## APLICACAO WEB PARA ANALISE DE DADOS E MACHINE LEARNING

Este projeto e o trabalho final da disciplina, com o objetivo de criar uma aplicacao web em Python que permite ao usuario fazer upload de um arquivo .csv, visualizar analises de dados e realizar predicoes com machine learning.

Esta e a estrutura base (backend) da aplicacao, desenvolvida com Django.

#### TECNOLOGIAS

- Python

- Django

- Pandas

### ESTADO ATUAL

O backend da aplicação está estruturado com Django e já conta com as seguintes funcionalidades:

- **Upload de Arquivos**: Sistema para upload de arquivos `.csv` via interface web.
- **Processamento de Dados**: Após o upload, os dados são lidos com Pandas e armazenados na sessão do usuário.
- **Análise Automática**: A classe `DataAnalyzer` realiza a limpeza dos dados, identifica os tipos de colunas (numéricas, categóricas, temporais, etc.) e gera um conjunto completo de visualizações, incluindo:
    - Análises univariadas e bivariadas (histogramas, boxplots, heatmaps).
    - Análises temporais e geográficas.
- **Estrutura Web**: URLs e templates básicos para as páginas de Upload, Análise e Predição estão configurados.

### COMO RODAR O PROJETO

O Python e o 'pip' devem estar instalados.

Clone o repositorio.

```python -m venv venv```

(Linux/Mac) ```source venv/bin/activate```

(Windows) ```venv\Scripts\activate```

```pip install -r requirements.txt```

Navega ate a pasta 'trabalhofinal' (onde esta o manage.py).

```python manage.py migrate```

```python manage.py runserver```
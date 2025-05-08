# Projeto de Classificação de Imagens Médicas com Arquiteturas 2D e 3D

Este projeto tem como objetivo realizar a classificação de imagens médicas utilizando redes neurais baseadas em arquiteturas Vision Transformer (ViT), tanto para entradas 2D quanto 3D. A estrutura foi desenvolvida com flexibilidade para adaptar diferentes tipos de dados de entrada e estratégias de treinamento.

## 🧠 Arquiteturas Suportadas
- **ViT 2D**: Para imagens planas (como cortes de ressonância).
- **ViT 3D**: Para volumes completos (como séries volumétricas de exames médicos).

## 📁 Estrutura Atual
- `configs/`: Arquivos YAML com hiperparâmetros e configurações.
- `data/`: Dados de treino e teste (.parquet).
- `logs/`: Logs e métricas visuais, como matrizes de confusão.
- `models/`: Pesos treinados (.pth).
- `scripts/`: Scripts auxiliares, como visualização de imagens.
- `src/`: Código-fonte principal com carregamento de dados, definição de modelo, treinamento e avaliação.

## ▶️ Como Executar

1. Instale as dependências:
   ```bash
   pip install -r requirements.txt

2. Ajuste as configurações em configs/config.yaml.

3. Inicie o treinamento:
    ```bash
    python3 scripts/run_training.py

3. Para visualizar amostras dos dados:
    ```bash
    python scripts/test_view_images.py

## 🧪 Entrada Esperada

Os dados devem estar em formato .parquet contendo:

- image.bytes: imagem em bytes

- label: rótulo de classificação

- image.path (opcional): nome da imagem original

## 🔄 Futuras Extensões

# PRECISO CORRIGIR OS PATHS EM CADA .PY
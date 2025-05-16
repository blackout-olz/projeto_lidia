# Classificação de Imagens Médicas com Vision Transformers (2D e 3D)

Este projeto tem como objetivo realizar a **classificação automática de imagens médicas de ressonância magnética (MRI)** para auxiliar no **diagnóstico precoce de doenças neurodegenerativas (AD)**. Utiliza arquiteturas Vision Transformer (ViT), com suporte para dados em **2D (cortes)** e **3D (volumes completos)**. Atualmente, o foco está nas arquiteturas **2D**, com extensibilidade futura para o uso de ViTs 3D.

---

## Funcionalidades Implementadas

- Treinamento de cinco arquiteturas ViT 2D distintas usando `timm`
- Otimização de hiperparâmetros com **Optuna**
- Geração de relatórios PDF para **cada trial**, com:
  - Gráficos de acurácia e perda
  - Métricas como F1, AUC, precisão, etc.
  - Hiperparâmetros utilizados
- Salvamento automático dos modelos (.pth)
- Teste do melhor modelo no conjunto de teste
- Geração de resumo final (`.txt`) com as métricas da melhor trial

> A parte 3D está em fase de planejamento e será desenvolvida com lógica semelhante.

---

## Estrutura de Diretórios

```plaintext
configs/          # Arquivos YAML com configurações específicas para 2D e 3D
data/             # Conjunto de dados .parquet (2D e futuramente 3D)
logs/             # Logs e visualizações por trial (matrizes, gráficos etc)
models/           # Pesos treinados (.pth)
scripts/          # Scripts auxiliares de tuning, visualização, etc.
src/              # Código-fonte principal
│
├── common/               # Funções compartilhadas (avaliação, métricas etc.)
├── data_loader_2d/       # Dataset e loader para dados 2D
├── data_loader_3d/       # [em construção] Loader para dados 3D
├── model_2d/             # Modelos 2D (ViT via timm)
├── model_3d/             # [em construção] Suporte para ViT 3D
├── train.py              # Lógica de treinamento principal
```

## Arquiteturas ViT 2D Suportadas

Usando a função create_model() da biblioteca timm, as seguintes arquiteturas já estão disponíveis:

- vit_small_patch16_224
- crossvit_15_240
- levit_192
- deit_base_patch16_224
- swin_tiny_patch4_window7_224

> As ViTs 3D ainda serão definidas, mas o plano é aplicar abordagem semelhante (tunning + teste).

## Formato de Entrada Esperado

- Imagens 2D (3D no futuro). Módulo data_loader_* deve ser adaptado para cada tipo de imagem utilizada no treinamento. Atualmente é usado um dataset com um conjunto de imagens em .parquet, disponível em:
> https://www.kaggle.com/datasets/borhanitrash/alzheimer-mri-disease-classification-dataset

## Como Executar

### 1. Instalar as dependências
```bash
pip install -r requirements.txt
```

### 2. Configurar (não implementado atualmente)

Edite os arquivos em configs/config_2d.yaml ou configs/config_3d.yaml com os hiperparâmetros desejados.

### 3. Rodar tuning de uma arquitetura
```bash
python3 scripts/tuner.py
```
* Isso poderá ser pulado no futuro se optar por usar uma arquitetura já tunada e disponibilizada (ainda não implementado)

## Saídas geradas (treinamento)

Após a execução, você terá:
- Modelos salvos em models/
- Logs e visualizações em logs/
- Um .pdf para cada trial com detalhes completos
- Um .txt final com as métricas da melhor execução no conjunto de teste

## Planejamento futuro

- Treinamento e tuning de ViTs 3D
- Comparação entre arquiteturas 2D e 3D
- Aplicação web e/ou desktop com carregamento de imagens e classificação automática
- Avaliação robusta em novos datasets

## Tecnologias utilizadas

- Python
- PyTorch + timm
- Optuna
- Matplotlib, Seaborn, FPDF, sklearn, numpy..
- [futuramente] MONAI (provavelmente) para suporte 3D

##
Desenvolvido por André

Este projeto faz parte de uma pesquisa voltada para diagnóstico precoce de doenças neurodegenerativas (AD) com IA em imagens médicas (MRI).
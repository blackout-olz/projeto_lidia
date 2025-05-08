import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from PIL import Image
import io
import matplotlib.pyplot as plt

# Lê o parquet
df = pd.read_parquet('data/train.parquet')

# Verifica as colunas
print("Colunas:", df.columns)

# Seleciona uma imagem e converte de bytes para PIL
row = df.iloc[0]
img_bytes = row['image.bytes']  # <- Aqui está a correção
label = row['label']

from PIL import Image
import io
import matplotlib.pyplot as plt

image = Image.open(io.BytesIO(img_bytes))

# Mostra imagem
plt.imshow(image, cmap='gray')
plt.title(f'Classe: {label}')
plt.axis('off')
plt.show()

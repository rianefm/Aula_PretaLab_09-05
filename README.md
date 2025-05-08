
# 📷 Classificação de Imagens com Transformers e CNNs

Este projeto realiza **classificação de imagens** usando **modelos pré-treinados da Hugging Face** e também demonstra como treinar um classificador simples com **PyTorch/TensorFlow**. A imagem utilizada como exemplo é baixada diretamente da internet e processada por modelos como `ViT` (Vision Transformer), `DETR` (para detecção de objetos) e `MobileNetV2` (para extração de features).

---

## 🧠 Modelos Utilizados

- `google/vit-base-patch32-384` – Classificação de imagens com Vision Transformers.
- `facebook/detr-resnet-50` – Detecção de objetos na imagem.
- `google/vit-base-patch16-224` – Classificação adicional da imagem.
- `MobileNetV2` – Extração de características visuais para treinar um classificador customizado.

---

## 📁 Etapas do Projeto

### 1. Download da Imagem

A imagem é baixada diretamente de um link público do Freepik:

```bash
wget https://img.freepik.com/fotos-gratis/jovem-mulher-negra-surpresa-com-a-boca-aberta_23-2148183287.jpg
```

---

### 2. Classificação com Vision Transformer

Utiliza-se o pipeline de classificação com modelo pré-treinado:

```python
from transformers import pipeline
from PIL import Image

pipe = pipeline("image-classification", model="google/vit-base-patch32-384", device=0)
imagem = Image.open("jovem-mulher-negra-surpresa-com-a-boca-aberta_23-2148183287.jpg")
res = pipe(imagem)
```

Resultado esperado:

```
🏷️ WIG                            | 📊 90.79%
🏷️ MASK                           | 📊 0.45%
🏷️ SOMBRERO                       | 📊 0.35%
🏷️ MARACA                         | 📊 0.31%
🏷️ HAIR SPRAY                     | 📊 0.20%
```

---

### 3. Detecção de Objetos com DETR

```python
detector = pipeline("object-detection", model="facebook/detr-resnet-50")
detections = detector(imagem)
```

Filtragem do objeto principal (por exemplo, uma pessoa):

```python
person_bbox = [d for d in detections if d['label'] == 'person'][0]
```

---

### 4. Classificação com outro modelo (para validação)

```python
classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
result = classifier(imagem)
```

---

### 5. Treinamento de Modelo Customizado

Utiliza-se TensorFlow para treinar um classificador a partir de um dataset customizado:

```bash
pip install torch torchvision matplotlib
```

Resultados de treinamento (exemplo):

```
Epoch 15/15 - accuracy: 0.8521 - val_accuracy: 0.7500
```

---

## 📦 Tecnologias

- Python 3.8+
- Hugging Face Transformers
- PIL (Pillow)
- TensorFlow ou PyTorch
- Matplotlib (opcional para visualização)
- torchvision (para datasets ou transformações)

---

## 🚀 Como Rodar

1. Clone o projeto:
```bash
git clone https://github.com/seu-usuario/classificador-imagens.git
cd classificador-imagens
```

2. Instale os pacotes:
```bash
pip install -r requirements.txt
```

3. Execute o notebook ou script principal:
```bash
python classificar.py
```

---

## 📚 Objetivo

Explorar técnicas modernas de **visão computacional** usando redes neurais pré-treinadas e analisar imagens com classificadores e detectores de objetos, além de treinar um modelo simples com transferência de aprendizado.

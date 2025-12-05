[# 🧠 RNN Simple - Operaciones Matri.txt](https://github.com/user-attachments/files/23964838/RNN.Simple.-.Operaciones.Matri.txt)
# 🧠 RNN Simple - Operaciones Matriciales en IA

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-orange)](https://numpy.org/)

## 📌 ¿Qué hace este código?

Este código muestra **cómo funcionan las Redes Neuronales Recurrentes (RNN)** usando solo operaciones matriciales con NumPy. Es una versión simplificada de lo que usan sistemas como:

- 🤖 **Chatbots** (Siri, Alexa)
- 🔤 **Traductores automáticos**
- 📈 **Análisis de series temporales**

## 🎯 ¿Por qué matrices?

| Operación | ¿Qué hace? | ¿Por qué es clave? |
|-----------|------------|-------------------|
| `Wx · x_t` | Transforma la entrada actual | Convierte datos en "features" útiles |
| `Wh · h_prev` | Combina con memoria pasada | ¡Esto da "memoria" a la red! |
| `tanh()` | Regula el resultado | Evita valores explosivos |

## 🚀 Cómo usar

```bash
# 1. Clona el repo
git clone https://github.com/tu-usuario/rnn-simple.git

# 2. Instala dependencias
pip install numpy

# 3. Ejecuta
python simple_rnn.py

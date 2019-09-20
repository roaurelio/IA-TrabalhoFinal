# Classificador de imagens de fundo de olho

O trabalho a ser apresentado irá utilizar um classificador de imagens e, a partir do treinamento feito numa base de dados irá classificar se a imagem é de fundo de olho ou não. 

O classificador utilizado será o deep learning, que é um algoritmo de aprendizado de máquina supervisionado cuja estrutura é inspirada no cérebro humano. Sua arquitetura possui redes neurais artificiais para aprender padrões através de tarefas repetidas durante seu treinamento. O classificador foi implementado na liguagem python, e foram utilizadas as bibliotecas TensorFlow e Keras. O código do algoritmo encontra-se no diretório src, no repositório do projeto.

Dois algoritmos serão utilizados, o primeiro será um deep learning sem camadas convolucionais e outro será utilizando convolução. O modelo sem convolução tem cinco camadas e o modelo com convolução tem quatro camadas de convolução, seguidas de uma de maxpooling e um bloco totalmente conectado (rede neural normal).

As imagens de fundo de olho utilizadas foram encontradas em bases de dados públicas disponibilizadas na internet. A base de dados que contém imagens que não são de fundo de olho contém imagens da parte externa do olho humano em várias situações (aberto, fechado, etc) e também é uma base online pública.

Bibliotecas necessárias para rodar os classificadores:
  * TensorFlow;  
  * Numpy;
  * Keras;
  * Matplotlib.


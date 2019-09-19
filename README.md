# Classificador de imagens de fundo de olho

O trabalho a ser apresentado irá utilizar um classificador de imagens e, a partir do treinamento feito numa base de dados irá classificar se a imagem é de fundo de olho ou não. 

O classificador utilizado será o deep learning, que é um algoritmo de aprendizado de máquina supervisionado cuja estrutura é inspirada no cérebro humano. Sua arquitetura possui redes neurais artificiais para aprender padrões através de tarefas repetidas durante seu treinamento. O classificador será implementado na liguagem python, e foram utilizadas as bibliotecas TensorFlow e Keras. O código do algoritmo encontra-se no repositório do projeto.

As imagens de fundo de olho utilizadas foram encontradas nas bases de dados online disponibilizada pelo DRIVE (Digital Retinal Images for Vessel Extraction) e pelo DRIONS-DB (Digital Retinal Images for Optic Nerve Segmentation Database), que também estarão no repositório do projeto. A base de dados que contém imagens que não são de fundo de olho contém imagens que possibilitam o diagnóstico de um paciente com cancêr e também é uma base online pública.

Bibliotecas necessárias:
  * TensorFlow;  
  * Numpy;
  * Keras;
  * Matplotlib

Através da curva ROC, que é uma métrica muito utilizada para avaliação de modelos de aprendizado de máquina, espera-se obter o quão eficiente foi o modelo, ao observar os parâmetros de falso positivo e verdadeiro positivo.

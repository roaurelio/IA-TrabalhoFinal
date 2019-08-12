# Classificador de imagens de fundo de olho

O trabalho a ser apresentado irá utilizar um classificador de imagens e, a partir do treinamento feito num banco de dados irá classificar se a imagem é de fundo de olho ou não. 

O classificador utilizado será o KNN (K - Nearest Neighbors), pois a base de dados utilizada é pequena. O classificador é um algoritmo de aprendizado de máquina supervisionado e baseia-se no quão semelhante é um dado do outro. O treinamento é formado por vetores de dimensão. O classificador será implementado na liguagem python, e seu algoritmo será colocado no repositório do projeto.

A base de dados será a base online disponibilizada pelo DRIVE (Digital Retinal Images for Vessel Extraction) e pelo DRIONS-DB (Digital Retinal Images for Optic Nerve Segmentation Database), que também estarão no repositório do projeto.

Metas a serem alcançadas:

1. Baixar bases de dados;
2. Implementar algoritmo do classificador;
3. Treinar classificador;
4. Realizar testes e colher resultados;
5. Fazer análise dos resultados.

Através da curva ROC, que é uma métrica muito utilizada para avaliação de modelos de aprendizado de máquina, espera-se obter o quão eficiente foi o modelo, ao observar os parâmetros de falso positivo e verdadeiro positivo.

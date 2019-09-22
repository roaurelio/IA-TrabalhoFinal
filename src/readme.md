Este diretório possui o código fonte dos algoritmo CNN e Deep Learning utilizados para classificar se a imagem é de fundo de olho ou não. Os algoritmos estão no formato .py e no formato .ipynb.
Para a execução dos algoritmos é preciso instalar as seguintes bibliotecas:

* TensorFlow;
* Numpy;
* Keras;
* Matplotlib;
* Opencv.

Duas variáveis deverão ser atualizadas para a correta execução dos algoritmos:
* data_dir
* data_dir_test 

As imagens representam o caminho o qual o programa irá buscar as imagens de treinamento e as imagens de teste.
O programa espera a seguinte estrutura de diretórios:
Um diretório principal "data" onde o mesmo contém dois subdiretórios: "test" e "train". Estes subdiretórios devem conter duas imagens representando as possíveis classes, que são "fundusImage" e "other", que são as imagens de fundo de olho e outras imagens, respectivamente.
Segue abaixo estrutura dos diretórios:

* data
    * train
      * fundusImage
      * other
    * test
      * fundusImage
      * other

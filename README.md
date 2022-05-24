## ViT
an implementation of the paper [Vision Transformer for Small-Size Datasets](https://arxiv.org/abs/2112.13492)
for running an instance -
1.    clone the repository
      ```
      git clone https://github.com/KaranjotSV/ViT.git
      cd ViT
      ```
      change the current working directory to ViT
2.    create a virtual environment
      ```
      virtualenv env
      ```
      a virtual environment named 'env' will be created
3.    activate the environment
      ```
      source env/bin/activate
      ```
4.    install the requirements
      ```
      pip install -r requirements.txt
      ```
5.    run
      ```
      python3 run.py
      ```

performance of ViT extended with SPT and LSA on CIFAR-100, trained for 20 epochs

| model  | top-1 accuracy (%) |
|--------|--------------------|
| ViT    |        45.05       |
| T-ViT  |        45.92       |
| M-ViT  |        44.99       |
| L-ViT  |        46.01       |
| S-ViT  |        47.39       |
| SL-ViT |        47.98       |

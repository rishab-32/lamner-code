# LAMNER-Code


LAMNER-Code employs semantic-syntactic embeddings for code comment generation.
# Download Dataset

  - The souce of dataset is https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text
  - Pre-processed dataset and trained embeddings used in the paper can be downloaded from https://drive.google.com/drive/folders/1I1uL4LXtagNSXSXaW51QEks763N3R2M3?usp=sharing

### Training the models
1. Download the dataset
2. Dowload pre-trained embeddings

```sh
$ python lamner.py 
$ python decode-lamner-beam-act.py or python decode-beam1.py 
```

License
----
MIT

References
----

www.pytorch.org

https://github.com/bentrevett/pytorch-seq2seq.git

https://github.com/Maluuba/nlg-eval.git

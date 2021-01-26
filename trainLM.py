import flair.datasets
from flair.data import Dictionary
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus

def main():
    corpus = flair.datasets.UD_ENGLISH()
    is_forward_lm = True

    dictionary: Dictionary = Dictionary.load('chars')

    corpus = TextCorpus(r'corpus',
                        dictionary,
                        is_forward_lm,
                        character_level=True)
                        
    language_model = LanguageModel(dictionary,
                                   is_forward_lm,
                                   hidden_size=256,
                                   nlayers=1)

    # train language model
    trainer = LanguageModelTrainer(language_model, corpus)

    trainer.train('resources/taggers/code_language_model',#path to dataset to train language model
                  sequence_length=250,
                  mini_batch_size=100,
                  max_epochs=50)

if __name__ == '__main__':
    main()

from model.txt_embeddings import WeTokenizer
from transformers import BertTokenizer


def create_tokenizer(tokenizer_type):
  """Creates a tokenizer given a tokenizer type."""
  if tokenizer_type.endswith('frz'):
    freeze = True
  elif tokenizer_type.endswith('ftn'):
    freeze = False
  if tokenizer_type.startswith('bert'):
    model_name_or_path = 'bert-base-cased'
    do_lower_case = True
    cache_dir = 'data/cache_dir'
    tokenizer_class = BertTokenizer
    tokenizer = tokenizer_class.from_pretrained(model_name_or_path,
                                                do_lower_case=do_lower_case,
                                                cache_dir=cache_dir)
  elif tokenizer_type.startswith('wo2v'):
    we_filepath = 'data/word_embeddings/word2vec/GoogleNews-vectors-negative300.bin'
    tokenizer = WeTokenizer(we_filepath, freeze=freeze)
  elif tokenizer_type.startswith('grvl'):
    we_filepath = 'data/word_embeddings/GrOVLE/mt_grovle.txt'
    tokenizer = WeTokenizer(we_filepath, freeze=freeze)
  else:
    tokenizer = None

  return tokenizer

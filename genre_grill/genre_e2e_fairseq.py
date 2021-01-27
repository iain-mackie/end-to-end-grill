
from genre import GENRE
from genre.entity_linking import get_end_to_end_prefix_allowed_tokens_fn_fairseq
from genre_grill import genre_e2e_fairseq


def get_entities(sentences, model):
    """ """
    prefix_allowed_tokens_fn = get_end_to_end_prefix_allowed_tokens_fn_fairseq(model, sentences)

    return model.sample(
        sentences,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    )


def pre_process_text(s):
    """ """
    s = s.replace('\t', ' ').replace('\n', ' ')
    s = s.replace('{', '(').replace('}', ')').replace('[', '(').replace('}]', ')')
    return " ".join(s.split())


def get_process_documents():
    return {}

if __name__ == '__main__':

    genre_path = '/Users/iain/LocalStorage/coding/github/end-to-end-grill/genre_grill/data/fairseq_e2e_entity_linking_aidayago'
    # genre_path = '/nfs/trec_robust04/end-to-end-grill/genre_grill/data/fairseq_e2e_entity_linking_aidayago'

    genre_model = GENRE.from_pretrained(genre_path, eval_bleu_args='', eval_bleu_detok_args='').eval()






from genre import GENRE
from genre.entity_linking import get_end_to_end_prefix_allowed_tokens_fn_fairseq


def get_entities(sentences, model):
    """ """
    prefix_allowed_tokens_fn = get_end_to_end_prefix_allowed_tokens_fn_fairseq(model, sentences)

    return model.sample(
        sentences,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    )




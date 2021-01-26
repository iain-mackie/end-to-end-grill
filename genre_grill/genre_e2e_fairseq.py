

if __name__ == '__main__':
    from genre import GENRE
    path = '/Users/iain/LocalStorage/coding/github/end-to-end-grill/genre_grill/data/fairseq_e2e_entity_linking_aidayago'
    model = GENRE.from_pretrained(path).eval()
    from genre.entity_linking import get_end_to_end_prefix_allowed_tokens_fn_fairseq

    sentences = ["In 1921, Einstein received a Nobel Prize."]

    prefix_allowed_tokens_fn = get_end_to_end_prefix_allowed_tokens_fn_fairseq(model, sentences)

    model.sample(
        sentences,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    )

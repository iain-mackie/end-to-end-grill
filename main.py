from transformers import BartForConditionalGeneration, BartTokenizer

import pickle

if __name__ == '__main__':


    e2e_path = '/Users/iain/LocalStorage/coding/github/end-to-end-grill/genre_grill/data/hf_e2e_entity_linking_wiki_abs'
    ed_path = '/Users/iain/LocalStorage/coding/github/end-to-end-grill/genre_grill/data/hf_entity_disambiguation_blink'
    kilt_path = '/Users/iain/LocalStorage/coding/github/end-to-end-grill/genre_grill/data/kilt_titles_trie.pkl'

    tokenizer = BartTokenizer.from_pretrained(ed_path)
    model = BartForConditionalGeneration.from_pretrained(ed_path).eval()

    sentences = ["[START_ENT] Armstrong [END_ENT] was the first man on the Moon."]

    input_args = {
        k: v.to(model.device) for k, v in tokenizer.batch_encode_plus(
            sentences,
            padding=True,
            return_tensors="pt"
        ).items()
    }


    with open(kilt_path, "rb") as f:
        trie = pickle.load(f)


    def prefix_allowed_tokens_fn(batch_id, sent):
        return trie.get(sent.tolist())

    print(tokenizer.batch_decode(
        model.generate(
            **input_args,
            min_length=0,
            num_beams=5,
            num_return_sequences=5,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        ),
        skip_special_tokens=True
    ))
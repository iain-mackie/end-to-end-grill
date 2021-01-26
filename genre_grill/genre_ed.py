from transformers import BartForConditionalGeneration, BartTokenizer
import pickle


def get_top_ed_entities(sentences, model, tokenizer, trie, num_beams=5, num_return_sequences=5, min_length=0):
    """ TODO"""

    input_args = {
        k: v.to(model.device) for k, v in tokenizer.batch_encode_plus(
            sentences,
            padding=True,
            return_tensors="pt"
        ).items()
    }

    def prefix_allowed_tokens_fn(batch_id, sent):
        return trie.get(sent.tolist())

    return tokenizer.batch_decode(
        model.generate(
            **input_args,
            min_length=min_length,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        ),
        skip_special_tokens=True
    )

if __name__ == '__main__':

    ed_path = '/Users/iain/LocalStorage/coding/github/end-to-end-grill/genre_grill/data/hf_entity_disambiguation_blink'
    kilt_path = '/Users/iain/LocalStorage/coding/github/end-to-end-grill/genre_grill/data/kilt_titles_trie.pkl'

    with open(kilt_path, "rb") as f:
        trie = pickle.load(f)

    tokenizer = BartTokenizer.from_pretrained(ed_path)
    model = BartForConditionalGeneration.from_pretrained(ed_path).eval()

    # sentences = ["[START_ENT] Armstrong [END_ENT] was the first man on the Moon.",
    #              "Armstrong was the first man on the [START_ENT] Moon [END_ENT]."]

    sentences = ["""Although the relations between France and
    the countries of the South Pacific have improved thanks to the
    moratorium on [START_ENT] nuclear testing [END_ENT] declared by the president of the
    Republic, the recurring debate on whether or not it is necessary
    to restart explosions and the possibility of a decision running
    counter to that made by Francois Mitterrand after the next""",
     """Celtic paid around £300,000 for Frimpong, with Lennon refusing to 
     divulge the identity of the bidding [START_ENT] club [END_ENT] and saying he did not know if 
     Manchester City had negotiated a percentage of any future sell-on fee."""
     ]

    entities = get_top_ed_entities(sentences, model, tokenizer, trie)
    print(entities)


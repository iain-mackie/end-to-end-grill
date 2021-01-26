from transformers import BartForConditionalGeneration, BartTokenizer
from genre.entity_linking import get_end_to_end_prefix_allowed_tokens_fn_hf


if __name__ == '__main__':

    e2e_path = '/Users/iain/LocalStorage/coding/github/end-to-end-grill/genre_grill/data/hf_e2e_entity_linking_wiki_abs'

    tokenizer = BartTokenizer.from_pretrained(e2e_path)
    model = BartForConditionalGeneration.from_pretrained(e2e_path).eval()

    sentences = ["In 1921, Einstein received a Nobel Prize."]
    # sentences = ["Armstrong was the first man on the Moon."]


    input_args = {
        k: v.to(model.device) for k, v in tokenizer.batch_encode_plus(
            sentences,
            return_tensors="pt"
        ).items()
    }
    prefix_allowed_tokens_fn = get_end_to_end_prefix_allowed_tokens_fn_hf(tokenizer, sentences)

    output = tokenizer.batch_decode(
        model.generate(
            **input_args,
            min_length=0,
            num_beams=5,
            num_return_sequences=5,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        ),
        skip_special_tokens=True
    )
    for i in output:
        print(i)


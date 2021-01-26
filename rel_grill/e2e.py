
from REL.mention_detection import MentionDetection
from REL.entity_disambiguation import EntityDisambiguation
from REL.ner import load_flair_ner

#import spacy


def get_REL_models(rel_base_url, rel_wiki_year):
    """ """
    rel_model_path = "{}/ed-wiki-{}/model".format(rel_base_url, rel_wiki_year)
    wiki_version = "wiki_" + str(rel_wiki_year)

    mention_detection = MentionDetection(rel_base_url, wiki_version)
    tagger_ner = load_flair_ner("ner-fast")
    config = {
        "mode": "eval",
        "model_path": rel_model_path,
    }
    entity_disambiguation = EntityDisambiguation(rel_base_url, wiki_version, config)
    return mention_detection, tagger_ner, entity_disambiguation

def get_entity_ranking_REL(text, mention_detection, tagger_ner, entity_disambiguation):
    """ Get entity ranking using REL. """
    input_text = {'0': [text, []]}
    # nlp = spacy.load("en_core_web_sm")
    # doc = nlp(text)
    # input_text = {str(i): [sent.text, []]for i, sent in enumerate(doc.sents)}

    mentions_dataset, n_mentions = mention_detection.find_mentions(input_text, tagger_ner)
    predictions, timing = entity_disambiguation.predict(mentions_dataset)

    return predictions


if __name__ == '__main__':

    query = '620'
    rel_base_url = '/Users/iain/LocalStorage/coding/github/end-to-end-grill/rel_grill/data/'
    rel_wiki_year = '2019'
    text = 'This is a test for linking animals and presidents of the USA.'
    mention_detection, tagger_ner, entity_disambiguation = get_REL_models(rel_base_url, rel_wiki_year)
    predictions = get_entity_ranking_REL(text, mention_detection, tagger_ner, entity_disambiguation)
    print(predictions)
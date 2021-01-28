
from REL.mention_detection import MentionDetection
from REL.entity_disambiguation import EntityDisambiguation
from REL.ner import load_flair_ner
import sys
import json
import os

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

def get_entity_ranking_REL(input_json, mention_detection, tagger_ner, entity_disambiguation):
    """ Get entity ranking using REL. """
    #input_text = {'0': [text, []]}
    # nlp = spacy.load("en_core_web_sm")
    # doc = nlp(text)
    # input_text = {str(i): [sent.text, []]for i, sent in enumerate(doc.sents)}
    processed_json = {doc_id: [text,[]] for doc_id, text in input_json.items()}

    mentions_dataset, n_mentions = mention_detection.find_mentions(processed_json, tagger_ner)
    predictions, timing = entity_disambiguation.predict(mentions_dataset)
    
    for 

    return predictions


if __name__ == '__main__':

    #assuming running from end-to-end folder
    rel_base_url = './rel_grill/data'
    rel_wiki_year = '2019'
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    
    for file in os.listdir(input_folder):
        if file[-5:] == ".json":
            with open(f"{input_folder}/{file}") as f:
                input_json = json.load(f)
            f.close()
            
            mention_detection, tagger_ner, entity_disambiguation = get_REL_models(rel_base_url, rel_wiki_year)
            predictions = get_entity_ranking_REL(input_json, mention_detection, tagger_ner, entity_disambiguation)
            
            with open(f"{output_folder}/{file[:-5]}_rel.json","w") as g:
                json.dump(predictions,g)
            g.close()
    
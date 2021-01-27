
from genre_grill.genre_e2e_fairseq import run_genre_e2e_linking

if __name__ == '__main__':
    from static import test_documents

    genre_path = './genre_grill/data/fairseq_e2e_entity_linking_aidayago'
    entity_links = run_genre_e2e_linking(documents=test_documents, model_path=genre_path, max_words=50, beam=5)
    print(entity_links)


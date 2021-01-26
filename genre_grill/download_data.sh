
wget http://dl.fbaipublicfiles.com/GENRE/hf_e2e_entity_linking_wiki_abs.tar.gz -P ./genre_grill/data/
wget http://dl.fbaipublicfiles.com/GENRE/kilt_titles_trie.pkl -P ./genre_grill/data/
wget http://dl.fbaipublicfiles.com/GENRE/hf_entity_disambiguation_blink.tar.gz -P ./genre_grill/data/
wget http://dl.fbaipublicfiles.com/GENRE/fairseq_e2e_entity_linking_wiki_abs.tar.gz -P ./genre_grill/data/
wget http://dl.fbaipublicfiles.com/GENRE/fairseq_e2e_entity_linking_aidayago.tar.gz -P ./genre_grill/data/

tar -xvf ./genre_grill/data/hf_e2e_entity_linking_wiki_abs.tar.gz -C ./genre_grill/data/
tar -xvf ./genre_grill/data/hf_entity_disambiguation_blink.tar.gz -C ./genre_grill/data/
tar -xvf ./genre_grill/data/fairseq_e2e_entity_linking_wiki_abs.tar.gz -C ./genre_grill/data/
tar -xvf ./genre_grill/data/fairseq_e2e_entity_linking_aidayago.tar.gz -C ./genre_grill/data/

rm -r -f ./genre_grill/data/hf_e2e_entity_linking_wiki_abs.tar.gz
rm -r -f ./genre_grill/data/hf_entity_disambiguation_blink.tar.gz
rm -r -f ./genre_grill/data/fairseq_e2e_entity_linking_wiki_abs.tar.gz
rm -r -f ./genre_grill/data/fairseq_e2e_entity_linking_aidayago.tar.gz

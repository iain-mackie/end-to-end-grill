
wget http://gem.cs.ru.nl/generic.tar.gz -P ./rel_grill/data/
wget http://gem.cs.ru.nl/wiki_2019.tar.gz -P ./rel_grill/data/
wget http://gem.cs.ru.nl/ed-wiki-2019.tar.gz -P ./rel_grill/data/

tar -xvf ./rel_grill/data/generic.tar.gz -C ./rel_grill/data/
tar -xvf ./rel_grill/data/wiki_2019.tar.gz -C ./rel_grill/data/
tar -xvf ./rel_grill/data/ed-wiki-2019.tar.gz -C ./rel_grill/data/

rm -r -f ./rel_grill/data/generic.tar.gz
rm -r -f ./rel_grill/data/wiki_2019.tar.gz
rm -r -f ./rel_grill/data/ed-wiki-2019.tar.gz
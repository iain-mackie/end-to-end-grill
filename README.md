# end-to-end-grill

# REL
pip install git+https://github.com/informagi/REL
python -m spacy download en_core_web_sm

# GENRE
git clone https://github.com/facebookresearch/GENRE/

cd GENRE/

pip install -r requirements.txt

python setup.py install 

git clone https://github.com/pytorch/fairseq 

cd fairseq 

on Linux: pip install --editable ./
on MacOS: CFLAGS="-stdlib=libc++" pip install --editable ./

# BLINK

git clone https://github.com/facebookresearch/BLINK

pip install -r requirements.txt

chmod +x download_blink_models.sh

./download_blink_models.sh


# Bootleg

# triplet-recsys

**Download data**

`sh download_data.sh`
- This will store original data files in `raw_data/`

**Prepare data**

`cd scripts`

`python prepare_data.py`
- This will save all required files in `saved/`
- 4 sub-categories are currently being used - Baby, Men, Women, Shoes

**Generate Metapaths for training skip-gram**

`cd scripts`

`python metapath_gen.py`
- This will generate a corpus of users and products and store it in a `.txt` file in `saved/`

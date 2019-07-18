# Transform predicted data into conll-format word by word file
# (not using dataframes immensely accelerates the process)
python /home/getalp/sfeirj/scripts/conlleval/preprocess_for_conlleval.py

# Evaluate via conll-eval
/home/getalp/sfeirj/scripts/conlleval/conlleval.pl < /home/getalp/sfeirj/data/valid_predictions_for_eval | head

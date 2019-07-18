# TEST THE CHANGE IN DATA GRANULARITY
# note that the doc_id itr starts with 0 and indicates docs in their original order in the data
for doc_id in range(len(train_firsts)):
    # find first and last sentence indices
    last_stc = -1 # idx of first sentence not to be considered
    first_stc = train_firsts[doc_id]
    if doc_id == len(train_firsts) - 1:
        last_stc = data_size
    else:
        last_stc = train_firsts[doc_id + 1]
    assert last_stc != -1
    # save span representations of the current document
    span_representations[first_stc:last_stc] = \
        change_dataset_granularity(data_size, args, trainer, task, epoch_itr, use_gold=True, use_attention=False)
print([len(s) for s in span_representations[:6]]) #nb of spans per sentence
print(span_representations[:6]) #actual numerical tensor representations

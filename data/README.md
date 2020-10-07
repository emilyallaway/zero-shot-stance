## VAST: Varied Stance Topics Data
New data released in this submission.
Short column descriptions
- author: username of the comment author
- post: original comment, unprocessed
- ori_topic: heuristically extracted topic
- ori_id: id generated to link post and heuristically extracted topics
- new_topic: updated topic from crowdsourced annotations
- label: stance label, 0=con, 1=pro, 2=neutral
- type_idx: type number, 1=HeurTopic, 2=CorrTopic, 3=ListTopic, 4=Synthetic neutral
- new_id: unique id for every comment-topic-label pair
- arc_id: id of the original article on NYT
- text: sentence and word tokenized and lowercased text, with punctuation and stopwords removed
- text_s: string version of text
- topic: tokenized and lowercased version topic, with punctuation and stopwords removed
- topic_str: string version of topic
- seen?: indicator for zero-shot or few-shot example, 0=zero-shot, 1=few-shot
- contains_topic?: indicator for whether topic is contained in the text, 0=no, 1=yes  
- change_lst: list of swapped words (unique to vast_test-sentswap.csv)  
- change_type: type of sentiment swapping  
- LexSim: a list of lexically similar training topics (if a zero-shot topic)  
- Qte: whether the example contains quotes (1=yes, 0=no)  
- Sarc: whether the example contains sarcasm (1=yes, 0=no)  
- Imp: whether the text contains the topic and the label is non-neutral (1=yes, 0=no)  
- mlS: whether there are other examples with the same document and different, non-neutral, stance labels (1=yes, 0=no)  
- mlT: whether there are other examples with the same document and different topics (1=yes, 0=no)

## MPQA
Existing sentiment lexicon. See mpqa/subjclueslen1-HLTEMNLP05.README

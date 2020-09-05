# Data for 'Zero-Shot Stance Detection: A Dataset and Model Using Generalized Topic Representations'
Submission to EMNLP 2020

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

## MPQA
Existing sentiment lexicon. See mpqa/subjclueslen1-HLTEMNLP05.README

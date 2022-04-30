import spacy
import json
from nltk.tag import pos_tag
from nltk.corpus import wordnet

def input_text(paper):
    text = ""
    data = json.load(paper)
    for d in data['body_text']:
        para=d['text']
        text += " " + para
    return text

def sentencize(text):
    sentencizer = spacy.load("en_core_web_sm")
    sentencizer.add_pipe("sentencizer")
    return [str(i) for i in sentencizer(text).sents],sentencizer

def load_ner_model(sentences,sentencizer):
    model = spacy.load('./model-last')
    index = 0
    sentences_after_ner = []
    for doc in model.pipe(sentences, disable=["tagger","parser"]):
        tokens_with_entities = []
        recognized_entities = []
        for ent in doc.ents:
            if " " in str(ent.text):
                sub_tokens = str(ent.text).split(" ")
                for sub_token in sub_tokens:
                    tokens_with_entities.append(sub_token)
                    recognized_entities.append((sub_token,str(ent.label_)))
        else:
            tokens_with_entities.append(str(ent.text))
            recognized_entities.append((str(ent.text),str(ent.label_)))
        final_entities = []
        for token in sentencizer(doc.text):
            if str(token) in tokens_with_entities and str(token) != ".":
                token_index = tokens_with_entities.index(str(token))
                final_entities.append(recognized_entities[token_index])
                del tokens_with_entities[token_index]
                del recognized_entities[token_index]
            else:
                final_entities.append((str(token),"O"))
        sentences_after_ner.append(tuple([sentences[index],final_entities]))
        index += 1
    return sentences_after_ner

def pos_tagging(sentences_after_ner):
    def get_token(sen_tup):
        return [i[0] for i in sen_tup[1]]
    sentences_for_pos = list(map(get_token,sentences_after_ner))
    for i in range(len(sentences_for_pos)):
        tagged_sentence = pos_tag(sentences_for_pos[i])
        pos_tagged_appended = []
        for j in range(len(sentences_after_ner[i][1])):
            pos_tagged_appended.append((sentences_after_ner[i][1][j][0],sentences_after_ner[i][1][j][1],tagged_sentence[j][1]))
        sentences_after_ner[i] = (sentences_after_ner[i][0],pos_tagged_appended)
    return sentences_after_ner

def synonym_substitution(sentences_after_ner):
    for i in range(len(sentences_after_ner)):
        final_sentence = sentences_after_ner[i][0]
        synonyms_substituted = []
        for token,entity,pos_tag in sentences_after_ner[i][1]:
            result = [token,entity,pos_tag]
            if entity == 'O' and pos_tag in ["RB","NN","VBN"] and len(list(wordnet.synsets(token))) > 0:
                result[0] = list(wordnet.synsets(token))[0].lemma_names()[0]
                final_sentence = final_sentence.replace(token,result[0])
            synonyms_substituted.append(tuple(result))
        sentences_after_ner[i] = (final_sentence,synonyms_substituted)
    return sentences_after_ner

# def entity_input():
#     entities = input("Enter a space-separated list of entities to look for: ")
#     entities = entities.upper()
#     relation = entities.split(" ")
#     return relation

def entity_counting(sentences_after_ner, relation):
    occurrences = []
    for sentence,entities in sentences_after_ner:
        relation_count = {}
        entity_sequence = [x[1] for x in entities]
        for entity in relation:
            relation_count[entity] = entity_sequence.count(entity)
        occurrences.append((sentence, sum(relation_count.values())))
    occurrences = sorted(occurrences,key = lambda x : -x[1])
    return occurrences

def sentence_ranking(sentences,occurrences):
    def get_sentence_from_score_tuple(t):
        return t[0]
    output_size = int(0.3 * len(sentences))
    output_sentences = map(get_sentence_from_score_tuple,occurrences[0:output_size])
    output_summary = ' '.join(output_sentences)
    return output_summary
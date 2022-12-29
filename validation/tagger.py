# The old_tagger.py starter code for CSC384 A4.
# Currently reads in the names of the training files, test file and output file,
# and calls the tagger (which you need to implement)

import os
import sys
import numpy as np


def file_reader(file_list):
    """Reads a list of files and returns a list of all sentences, a list of
        all corresponding tags and a list of all the distinct tags.
        len(sentences) == len(tags)"""
    sentences = []
    tags_list = []  # List of all tags categorized by sentences.
    distinct_tags = []  # List of all tags observed (one entry for each tag)
    for f in file_list:
        with open(f) as file:
            # 0 indicates that a sentence is in process of being added to the list
            sentence_flag = 0
            sentence = []
            tags = []  # All tags for this particular sentence.
            for line in file:
                if sentence_flag == 1:
                    sentence = []
                    tags = []
                    # indicates that a sentence is in process of being added to the list
                    sentence_flag = 0
                split_line = line.split(' : ')
                word_to_append = split_line[0]
                tag_to_append = split_line[1]
                word_to_append = word_to_append.strip()
                tag_to_append = tag_to_append.strip()
                if tag_to_append in ['AV0-AJ0', 'VVN-AJ0', 'VVD-AJ0', 'NN1-AJ0', 'VVG-AJ0', 'PRP-AVP',
                                     'CJS-AVQ', 'PRP-CJS', 'DT0-CJT', 'PNI-CRD', 'NP0-NN1', 'VVB-NN1',
                                     'VVG-NN1', 'VVZ-NN2', 'VVN-VVD']:
                    tag_to_append = tag_to_append[-3:] + '-' + tag_to_append[:3]
                sentence.append(word_to_append)
                tags.append(tag_to_append)
                if tag_to_append not in distinct_tags:
                    distinct_tags.append(tag_to_append)
                if word_to_append in ['.', '?', '!']:
                    # Indicates end of sentence. Start new sentence and append existing sentence.
                    sentences.append(sentence)
                    tags_list.append(tags)
                    # Indicates that a sentence has ended.
                    sentence_flag = 1
    return sentences, tags_list, distinct_tags


def test_file_reader(file):
    sentences = []
    with open(file) as file:
        # 0 indicates that a sentence is in process of being added to the list
        sentence_flag = 0
        sentence = []
        for line in file:
            if sentence_flag == 1:
                sentence = []
                # indicates that a sentence is in process of being added to the list
                sentence_flag = 0
            word_to_append = line.strip()
            sentence.append(word_to_append)
            if word_to_append in ['.', '?', '!']:
                # Indicates end of sentence. Start new sentence and append existing sentence.
                sentences.append(sentence)
                # Indicates that a sentence has ended.
                sentence_flag = 1
    return sentences


def initial_probability_table_generator(sentences, tags):
    """Creates a dictionary of initial probabilities given a list of sentences"""
    sentences_length = len(sentences)
    initial_probability_table = {}

    # counting number of first-word occurrences of each tag.
    for i in range(0, len(sentences)):
        relevant_tags_list = tags[i]
        first_tag = relevant_tags_list[0]
        if first_tag not in initial_probability_table:
            initial_probability_table[first_tag] = 1
        else:
            initial_probability_table[first_tag] += 1

    # converting tag observations to probability
    for ta in initial_probability_table:
        initial_probability_table[ta] = initial_probability_table[ta]/sentences_length

    return initial_probability_table


def emission_probability_table_generator(sentences, tags):
    """Creates a dictionary of dictionaries of possible emissions
        (i.e. words associated with tags)"""
    # Keys are the tags and the values are the number of words that have been tagged using the tag
    tag_counter = {}

    # Dict of dicts where the key is the tag and the values are dicts whose keys are the words
    # tagged using the tag and the value is the number of time such tag-word relationships
    # have been observed
    emission_probability_table = {}

    for i in range(0, len(sentences)):
        sentence = sentences[i]
        relevant_tags = tags[i]
        for j in range(0, len(sentence)):
            word = sentence[j]
            word_tag = relevant_tags[j]
            if word_tag not in tag_counter:
                tag_counter[word_tag] = 1
            else:
                tag_counter[word_tag] += 1

            if word_tag not in emission_probability_table:
                emission_probability_table[word_tag] = {word: 1}
            else:
                word_tag_word_dict = emission_probability_table[word_tag]
                if word not in word_tag_word_dict:
                    word_tag_word_dict[word] = 1
                else:
                    word_tag_word_dict[word] += 1

    # Converting observation counts to probability
    for emission_tag in emission_probability_table:
        word_dict = emission_probability_table[emission_tag]
        word_count = tag_counter[emission_tag]
        for word in word_dict:
            word_dict[word] = (word_dict[word])/word_count

    # for i in emission_probability_table:
    #     print(sum(emission_probability_table[i].values()))
    return emission_probability_table


def viterbi_trainer(sentences, tags):
    initial_probability_table = initial_probability_table_generator(sentences, tags)
    transition_probability_table = transition_probability_table_generator(sentences, tags)
    emission_probability_table = emission_probability_table_generator(sentences, tags)
    return initial_probability_table, transition_probability_table, emission_probability_table


def viterbi_calculator(sentences, distinct_tags, initial_table, transition_table, emission_table):
    """Runs the Viterbi Algorithm for each sentence in sentences."""
    predicted_tags = []
    for i in range(0, len(sentences)):
        sentence = sentences[i]
        prob = np.zeros((len(sentence), len(distinct_tags)))
        prev = []
        for n in range(0, len(sentence)):
            prev.append([])

        first_word = sentence[0]
        for j in range(0, len(distinct_tags)):
            # The way we are defining distinct tags, all tags in distinct tags will appear at least once in
            # the training set.
            possible_tag = distinct_tags[j]
            trained_words = emission_table[possible_tag]

            if first_word in trained_words:
                if possible_tag in initial_table:
                    prob[0, j] = initial_table[possible_tag] * trained_words[first_word]
                else:
                    # If the tag being investigated did not exist as a first word of a sentence in the training set,
                    # letting the probability be 0 since it is likely that this particular tag does not usually
                    # come at the beginning of a sentence.
                    prob[0, j] = 0
            else:
                prob[0, j] = 0
            prev[0].append(None)

        # if all elements of prob[0] are 0, it is likely that the first word was never trained on. In such a case,
        # we ignore the emission probability and just focus on the initial tag probability
        if np.all(prob[0] == 0):  # This means all elements of prob[0] are 0
            for j1 in range(0, len(distinct_tags)):
                possible_tag = distinct_tags[j1]

                if possible_tag in initial_table:
                    prob[0, j1] = initial_table[possible_tag]
                else:
                    prob[0, j1] = 0

        for k in range(1, len(sentence)):
            word = sentence[k]
            alt_prob_k_j = []
            for j in range(0, len(distinct_tags)):
                cur_tag = distinct_tags[j]
                prev_transition_product_list = []
                for ind in range(0, len(prob[k-1])):
                    active_tag = distinct_tags[ind]
                    transitions_second_tags = transition_table[active_tag]  # All transitions with active tag (one of the distinct tags) as the first tag
                    if cur_tag in transitions_second_tags:
                        to_append = prob[k-1][ind] * transitions_second_tags[cur_tag]
                    else:
                        to_append = 0
                    prev_transition_product_list.append(to_append)
                max_prev_transition = max(prev_transition_product_list)
                alt_prob_k_j.append(max_prev_transition)
                max_prob_index = prev_transition_product_list.index(max_prev_transition)
                # VV The tag which has the highest prob of causing the previous state and of getting the HMM to the current state VV
                max_prob_prev_state = distinct_tags[max_prob_index]
                prev[k].append(max_prob_prev_state)

                if word in emission_table[cur_tag]:
                    prob_k_j = max_prev_transition * emission_table[cur_tag][word]
                else:
                    prob_k_j = 0
                prob[k][j] = prob_k_j

            # if all elements of prob[k] are 0, then it is likely that the word we are considering was not seen in training.
            # In such a case, we ignore the emission probability and make our decision only based on the most likely previous tag
            # and the transition probability to the corresponding tag
            if np.all(prob[k] == 0):
                prob[k] = alt_prob_k_j

        predicted_tags.extend([viterbi_predictor(prob, prev, distinct_tags)])
    return predicted_tags


def transition_probability_table_generator(sentences, tags):
    """Creates a dictionary of probabilities of possible transitions
        (i.e. all transitions in the training file) """
    # Keys are the first tags in the transition and the values are the number of transitions with
    # the key as the first tag.
    transitions_dict = {}

    # Dict of dicts where the key of the first dict are first tags and the value is a dict whose
    # keys are the second tag in the transition and the value is the number of time such transitions
    # are seen.
    transition_probability_table = {}

    for i in range(0, len(sentences)):
        prev = ''
        sentence = sentences[i]
        relevant_tags = tags[i]
        for j in range(0, len(sentence)):
            word = sentence[j]
            if prev != '':
                prev_tag = relevant_tags[j - 1]
                curr_tag = relevant_tags[j]
                if prev_tag not in transitions_dict:
                    transitions_dict[prev_tag] = 1
                else:
                    transitions_dict[prev_tag] += 1

                if prev_tag not in transition_probability_table:
                    transition_probability_table[prev_tag] = {curr_tag: 1}
                else:
                    p_t_trans_prob = transition_probability_table[prev_tag]
                    if curr_tag not in p_t_trans_prob:
                        p_t_trans_prob[curr_tag] = 1
                    else:
                        p_t_trans_prob[curr_tag] += 1
            prev = word

    # converting observation counts to probability:
    for first_tag in transitions_dict:
        first_tag_transitions = transitions_dict[first_tag]
        second_tag_dict = transition_probability_table[first_tag]
        for second_tag in second_tag_dict:
            second_tag_dict[second_tag] = (second_tag_dict[second_tag])/first_tag_transitions

    return transition_probability_table

    # transitions = 0
    # transition_probability_table = {}
    # for i in range(0, len(sentences)):
    #     prev = ''
    #     sentence = sentences[i]
    #     relevant_tags = tags[i]
    #     for j in range(0, len(sentence)):
    #         word = sentence[j]
    #         if prev != '':
    #             prev_tag = relevant_tags[j-1]
    #             curr_tag = relevant_tags[j]
    #             if (prev_tag, curr_tag) not in transition_probability_table:
    #                 transition_probability_table[(prev_tag, curr_tag)] = 1
    #             else:
    #                 transition_probability_table[(prev_tag, curr_tag)] += 1
    #             transitions += 1
    #         prev = word
    #
    # for ta in transition_probability_table:
    #     transition_probability_table[ta] = transition_probability_table[ta]/transitions
    #
    # print(transition_probability_table)
    # return transition_probability_table


def viterbi_predictor(prob, prev, distinct_tags):
    predicted_tags = []
    i = len(prev) - 1
    if prev[i][0] is None:
        prev_index = np.argmax(prob[i])
        predicted_tag = distinct_tags[prev_index]
        predicted_tags.append(predicted_tag)
        return predicted_tags
    else:
        while prev[i][0] is not None:
            if i == (len(prev) - 1):
                prev_index = np.argmax(prob[i])
                predicted_tag = distinct_tags[prev_index]
                predicted_tags.append(predicted_tag)
                predicted_tags.append(prev[i][prev_index])
                i -= 1
            else:
                last_tag = predicted_tags[-1]
                prev_index = distinct_tags.index(last_tag)
                predicted_tags.append(prev[i][prev_index])
                i -= 1
        predicted_tags.reverse()
        return predicted_tags


def tag(training_list, test_file, output_file):
    # Tag the words from the untagged input file and write them into the output file.
    # Doesn't do much else beyond that yet.
    print("Tagging the file.")
    #
    # YOUR IMPLEMENTATION GOES HERE
    #
    sentences, tags, distinct_tags = file_reader(training_list)
    test_sentences = test_file_reader(test_file)
    initial_table, transition_table, emission_table = viterbi_trainer(sentences, tags)
    predicted_tags = viterbi_calculator(test_sentences, distinct_tags, initial_table, transition_table, emission_table)
    # print(predicted_tags)

    write_into(output_file, test_sentences, predicted_tags)
    return predicted_tags


def write_into(output_file, test_sentences, predicted_tags):
    file = open(output_file, "w")
    for i in range(0, len(test_sentences)):
        sentence = test_sentences[i]
        predicted_tags_sublist = predicted_tags[i]
        for j in range(0, len(sentence)):
            to_write = sentence[j] + ' : '
            to_write = to_write + predicted_tags_sublist[j] + '\n'
            file.write(to_write)


if __name__ == '__main__':
    # Run the tagger function.
    print("Starting the tagging process.")
    # s, t, tags_list_1 = file_reader('data/training1.txt')
    # initial_probability_table_generator(s, t)
    # transition_probability_table_generator(s, t)
    # emission_probability_table_generator(s, t)
    # tag(['data/training1.txt'], 'data/test1.txt', 'output_1.txt')

    # Tagger expects the input call: "python3 old_tagger.py -d <training files> -t <test file> -o <output file>"
    # TODO: Uncomment before submitting!!!!!!
    parameters = sys.argv
    training_list = parameters[parameters.index("-d")+1:parameters.index("-t")]
    test_file = parameters[parameters.index("-t")+1]
    output_file = parameters[parameters.index("-o")+1]
    # print("Training files: " + str(training_list))
    # print("Test file: " + test_file)
    # print("Output file: " + output_file)

    # Start the training and tagging operation.
    tag(training_list, test_file, output_file)

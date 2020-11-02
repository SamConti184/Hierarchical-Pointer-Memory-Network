import numpy as np
import random
from typing import Tuple, List

#Implementation of a class used to read the .txt Dataset and create a series
#of .npz files (serialized Numpy arrays) that will be used as starting point
#to create TensorFlow Datasets.
#The main objective of this class is the creation for each set (training, validation
#and testing) of 6 lists:
#1) dialogue_history in numerical IDs form (a collection of matrices, one matrix per complete dialogue)
#Notice that the dialogue history, for each dialogue, is composed by the user and sytem sentences as input.
#The last system response is contained in the output instead
#2) the effective length of each sentence in the dialogues
#3) dialogue_history in numerical IDs form with OOV words changed from 1 to a special ID
#4) dialogue history in textual form
#5) output (the labels) in textual form
#6) output in numerical form, with OOV words with specific ID
#7) output in numerical form, with OOV words with ID = 1
#These operations can be performed in anyway you prefer, even without this module

class Dataset:
    def __init__(self, dataset_name: str, kb_dropout: bool, dropout_probability: int, task: int = 5):
        self.PADDING = "PAD"  #Using 0 as a padding value is more suitable with tf.Keras
        self.word_to_id = {self.PADDING : 0, "UNK": 1, "<START>": 2, "<END>" : 3} #the dictionary used for converting words into IDs.
        self.oov_words_dict = {} #Dictionary of the oov_words in the current dataset
        self.dataset_name = dataset_name #this dataset is based for bAbI
        self.max_len_dialogue = 0
        self.task = task #bAbI has different tasks, this variable states which one (int )
        self.training = True #Modify if in training setting
        self.validation = True #Modify if in validating setting
        #The following attributes are used to contain the features, the outputs and
        #the OOV words taken from the current task.
        self.dialogue_history = [] #The dialogue history
        self.outputs = [] #Each output sentence for each dialogue history
        self.outputs_IDs = [] #Each output sentence, but in numerical ID form
        self.kb_dropout_candidates = set() #no duplicates needed
        self.kb_words = set() #The set of Knowledge Base words
        self.kb_dropout = kb_dropout #The percentage of words involved in Knowledge Dropout
        self.dropout_probability = dropout_probability #how sever is the dropout for the current task
        self.input_IDs = [] #List collecting the input in numerical form (OOV words with special IDs)
        self.input_text = [] #List collecting input in text form
        self.system_IDs = [] #List collecting the output in numerical form (OOV words with special IDs)
        self.labels = [] #List collecting the out in numerical form (OOV words with ID = 1)
        self.sentence_lenghts = []

    #Function used to create a new version of the file passed as argument where
    #the text will be properly pre-processed.
    #The pre-processing is based on dividing the input sentences in an input-output
    #structure and then on converting each word into its corresponding integer id of
    #the dataset dictionary.
    def create_preprocessed_file(self, filename):
        with open("./Datasets/Originals" + filename, "r") as source:
            lines = source.readlines()
        num_lines = len(lines)
        max_lens= self.getMaxLineLengths(lines)
        self.max_len_dialogue = max_lens[2] #saving the maximum length for a dialogue
        print("\nMax length: " + str(max_lens[0]) + " " + str(max_lens[1]) + " " + str(max_lens[2]))

        temp = [] #array containing the numerical IDs of each sentence
        temp_input_text = [] #array containing the text of each sentence
        temp_labels = [] #array containing the numerical IDs of the labels
        #Loop mainly based on the bAbI dataset .txt structure.
        for i in range(num_lines):
            line = lines[i]
            if(len(line) > 1): #if the next line is not empty (the current dialogue is not over)
                nextline = lines[i+1]
                if (line.split()[2][:2] == "R_"): #if line contains KB entity
                    #we will append entire KB entities porton in a single unit of dialogue
                    #The dataset information will be included in a single input sample.
                    dataset_line, line = self.parse_dataset_line(line, max_lens[0])
                    temp.append(dataset_line)
                    temp_input_text.extend(line)
                else:
                    #user_modified, system_modified, user, system, user_oov_words, system_oov_words, labels_IDs
                    user_modified, system_modified, user, system, labels_id = self.parse_dialogue_line(line, max_lens)
                    temp.append(user_modified) #Appending to the temp dialogue history the IDs of the user sentence
                    temp_input_text.extend(user) #Extending the input text with the textual user sentence
                    self.labels.append(labels_id) #Appending the labels array with the current dialogue's labels IDs
                    self.dialogue_history.append(temp[:]) #Appending to the current dialogue history
                    self.input_text.append(temp_input_text[:]) #Appending text to the current dialogue history
                    self.outputs.append(system[:-2]) #Appending the textual output
                    if(len(nextline) > 1): #if the dialogue is not over
                        temp.append(system_modified) #the next batch sample will have the system sentence as a list of IDs in the dialogue history.
                        temp_input_text.extend(system[1:-3] + system[-2:]) #Extending the text of the dialogue with the system sentence
            else:
                #resetting all the temp variables
                temp = []
                temp_input_text = []
                temp_labels = []
                #Clearing the kb_dropout candidates for the next dialogue
                self.kb_dropout_candidates.clear()

        #Creation of the numerical version of features and labels with appropriate numbering for OOV words
        self.input_IDs, self.system_IDs = self.create_new_input_IDs(self.dialogue_history, self.input_text, self.outputs, self.labels)
        self.set_sentences_lengths()
        destination = "./Datasets/Preprocessed" + filename
        np.savez(destination, x1=self.dialogue_history, x2=self.sentence_lenghts, x3=self.input_IDs, x4=self.input_text, y1=self.outputs, y2=self.system_IDs, y3=self.labels)
        #max len dialogue including the PADDING
        self.max_len_dialogue = max([len(dialogue) for dialogue in self.input_IDs])
        #resetting the temp variables for the next set
        self.dialogue_history = []
        self.outputs = []
        self.input_IDs = []
        self.system_IDs = []
        self.labels = []
        self.sentence_lenghts = []
        self.input_text = []

    #Get per each sentences the index of the last non-paddding word
    def set_sentences_lengths(self) -> None:
        res = []
        for mat in self.dialogue_history:
            temp = []
            for row in mat:
                temp.append(sum(1 for word in row if word != 0))
            self.sentence_lenghts.append(temp)


    #Creation of two lists:
    #input_IDs contains the numerical IDs of the input, only with special indices
    #for OOV words
    #system IDs contains the numerical IDs of the output, only with special indices
    #for OOV words
    def create_new_input_IDs(self, dialogue_history: List[List[List[int]]],
            input_text: List[List[str]], output: List[List[str]],
            labels: List[List[int]]) -> Tuple[List[List[int]], List[List[int]]]:
        input_IDs = []
        system_IDs = []
        idx = 0
        for dialogue, text, system_sample, label_sample in zip(dialogue_history, input_text, output, labels):
            #Flatten the dialogue to a 1-d array
            dialogue = [item for sublist in dialogue for item in sublist]
            for num, word_id in enumerate(dialogue):
                if (word_id == 1):
                    #If the word is OOV, set its ID to the len of the vocabulary + how many OOV were encountered so far
                    dialogue[num] = self.oov_words_dict[text[idx]] + len(self.word_to_id)
                if (word_id != 0):
                    idx += 1
            input_IDs.append(dialogue)
            res = []
            for word, label in zip(system_sample, label_sample):
                if word in self.oov_words_dict and label == 1:
                    res.append(self.oov_words_dict[word] + len(self.word_to_id))
                else:
                    res.append(label)
            system_IDs.append(res)
            idx = 0
        return input_IDs, system_IDs


    #Getting the maximum lengths of user and system lines for padding them
    #equally up to the longest sentence per each type.
    #Padding is performed in this phase and in Tensorflow, but could also
    #be performed entirely in TensorFlow.
    def getMaxLineLengths(self, lines: List[str]) -> List[int]:
        max_feature_len = max_target_len = max_dialogue_len = dialogue_len = 0
        for line in lines:
            #if the dialogue is not over
            if(len(line) > 1):
                #Updating the current dialogue length
                dialogue_len += len(line.split())
                #user and system lines always separated by a tabulation character
                sub_lines = line.split("\t")
                #Check the lenghts of all the sublines (user and system)
                #and the length only of the target sublines (system repsonse)
                if(len(sub_lines) > 1): #check if not a dataset line
                    target_len = len(sub_lines[1].split())
                    if (target_len > max_target_len):
                        max_target_len = target_len
                #Technically even the current target can become part of the
                #feature input later
                features_lengths = [len(line.split()) for line in sub_lines]
                if(max(features_lengths) > max_feature_len):
                    max_feature_len = max(features_lengths)
            else:
                if (dialogue_len > max_dialogue_len):
                    max_dialogue_len = dialogue_len
                dialogue_len = 0
        return [max_feature_len, max_target_len, max_dialogue_len]


    #Creating user and system sentences in numerical ID form, by tokenizing and
    #adding user and temporal information.
    #Also extraction of user and system sentences in textual form,
    #and creation of the output in numerical ID form.
    def parse_dialogue_line(self, line: str, max_len: int) -> Tuple[List[int], List[int], str, str, List[int]]:
        labels_IDs = []
        user, system = line.split("\t") #user and systems sentences are divided by a tabulation character
        n_id = user.split()[0] #getting the number of the line
        user = user[len(n_id)+1:]   #getting only the user line, without the number (one or two digits)
        user_modified = self.modify_line(user[:], n_id, "u", max_len[0])
        system_modified = self.modify_line(system[:], n_id, "s", max_len[0])
        system = "<START> " + system + " <END>"
        system_modified_no_pad = [id for id in system_modified if id != 0]
        labels_IDs.append(self.word_to_id["<START>"])
        labels_IDs.extend(system_modified_no_pad[:-2])
        labels_IDs.append(self.word_to_id["<END>"])
        system = system + " t" + str(n_id) + " $s"
        system = system.split()
        user = user + " t" + str(n_id) + " $u"
        user = user.split()
        #user_modified = the IDs version of the user's sentence, with padding, user and temp info
        #system_modified = the IDs version of the system's sentence, with padding, user and temp info
        #user and system = the textual version of the user and system sentences
        #labels_IDs = the system IDs version without padding or info about user and temp
        return user_modified, system_modified, user, system, labels_IDs

    #Parsing a line containing a KB tuple, usually result of a DB interrogation
    def parse_dataset_line(self, line: str, max_len: int) -> Tuple[str, str]:
        n_id = line.split()[0]
        line = line[len(n_id)+1:] #removing the initial number of the sentence
        dataset_line = self.modify_line(line[:], 1, "d", max_len)
        line = line.split()
        return dataset_line, line

    #Apply modifications to the original sentence: numerical IDs and user
    #and temporal information
    def modify_line(self, line: str , n_id: int, user: str, max_len: int) -> str:
        #If the line is a dataset line
        if(user != "d"):
            #Just Adding temporal and user information
            line = line + " t" + str(n_id) + " $" + user
        #If the line is a user sentence
        if(user == "u"):
            line = self.convert_to_ids(line, True, max_len)
        #If the line is a system sentence
        else:
            line  = self.convert_to_ids(line, False, max_len)
        return line

    #Converting each word of the sentence to the respective numerical ID in the
    #Dataset vocabulary
    def convert_to_ids(self, line: str, user: str, max_len: int, pad: bool =True) -> str:
        line = line.split()
        for i in range(len(line)):
            #Getting the numerical ID
            word_id = self.get_word_id(line[i])
            #Check if the word if OOV or if to make it OOV through
            #kb dropout feature
            if (self.condition_kb_dropout(line[i], user)):
                    word_id = 1
                    self.kb_dropout_candidates.add(line[i])
            if(word_id == 1):
                #If the word is not in the OOV dictionary, add to dict oov_words
                if(self.oov_words_dict.get(line[i], 1) == 1):
                    self.oov_words_dict[line[i]] = len(self.oov_words_dict)
            line[i] = word_id
        #Applying padding IDs to the line
        #The plus two is due to the two terms added (user and temporal info)
        if len(line) < max_len + 2 and pad:
            line += [self.word_to_id[self.PADDING]] * (max_len + 2 - len(line))
        return line

    #Get the word_id. If the word is not in the dictionary,
    #and we are in a training setting, add it to the dictionary and get its ID.
    def get_word_id(self, word: str) -> int:
        word_id = self.word_to_id.get(word, 1)
        if(self.training):
            if (word_id == 1):
                self.word_to_id[word] = len(self.word_to_id)
                word_id = self.word_to_id[word]
        return word_id

    #Verifying if KB dropout needs to be applied in the current scenario.
    #A word cna be transformed into a OOV word if it is an entity word (kb_words)
    #and if the same word is already been set to an OOV previously in the dialogue
    def condition_kb_dropout(self, word: str, user: str) -> bool:
        #if training phase or validation phase, if kb_dropout requested and if user sentence
        cond1 = (self.training or self.validation) and self.kb_dropout and user
        #if word inside set of kb_words (the set of words eligible to kb dropout) and probability successful
        cond2 = word in self.kb_words and random.random() < self.dropout_probability
        #check if word is already been chosen as a kb_dropout_candidate in the current dialogue
        cond3 = word in self.kb_dropout_candidates
        return (cond1 and cond2) or cond3


    #Creation of the so-called KB words. In original paper it is not given
    #a proper definition. KB words of bAbI are constituted not only by the named entities related
    #to each restaurant, such as name or address, but also common nouns
    #(e.g. the type of cuisine). In the dataset each KB word is followed or preceded
    #by a peculiar word (K_Type_of_KB_word).
    def create_kb_words(self, filename):
        with open("./Datasets/Originals" + filename, "r") as source:
            for line in source:
                line = line.split()
                #Checking if the line is a result from the KB, thus containing KB words:
                if (len(line) > 1 and len(line[2]) >= 2 and line[2][:2] == "R_"):
                    self.kb_words.add(line[1]) #Adding the name of the restaurant
                    self.kb_words.add(line[2]) #Adding the type of the entity
                    self.kb_words.add(line[3]) #Adding the value of the entity

    #Returning a list fo the three needed files for the specified dataset.
    #Task is an optional argument only used for bAbI dataset.
    def get_files_names(self, oov=False):
        if (self.dataset_name == "babi"):
            filename = "/dialog-bAbI-tasks/dialog-bAbI-task" + \
                        str(self.task) + "-" + self.babi_task_name(self.task) + "-"
        if(oov):
            test_filename = filename + "tst-OOV.txt"
        else:
            test_filename = filename + "tst.txt"
        train_filename = filename + "trn.txt"
        dev_filename = filename + "dev.txt"
        return train_filename, dev_filename, test_filename

    #A specific dictionary for gettin the proper name of the respective task
    #of the bAbI dataset
    def babi_task_name(self, task):
        return {1 : "API-calls",
                2 : "API-refine",
                3 : "options",
                4 : "phone-address",
                5 : "full-dialogs",
                6 : "dstc2"}.get(task, "dstc2")

    #The general process for properly pre-process a single task of the bAbI dataset.
    #This method will be called by external modules.
    def create_preprocessed_dataset(self, mode: str = "training", oov: bool = False):
        training, developing, testing = self.get_files_names(oov)
        if(mode == "training"):
            self.create_kb_words(training)
            self.create_preprocessed_file(training)
            #Following printing statements to check the OOV produced in the set
            print("train")
            print(len(self.oov_words_dict))
            print(self.oov_words_dict.keys())
        elif (mode == "develop"):
            self.training = False
            self.oov_words_dict.clear()
            self.create_preprocessed_file(developing)
            print("develop")
            print(len(self.oov_words_dict))
            print(self.oov_words_dict.keys())
        else:
            self.training = False
            self.validation = False
            self.oov_words_dict.clear()
            self.create_preprocessed_file(testing)
            #print(len(self.word_to_id))

    #Simple function returning desired properties of the dataset:
    #Here are returned he length of the vocabolary and of the number of
    #oov words (originals or produced by Knowledge Dropping) found
    #in the current dataset.
    def get_important_dimensions(self) -> Tuple[int, int]:
        return len(self.word_to_id), len(self.oov_words_dict)

if __name__ == "__main__":
        dataset = Dataset("babi", True, 0.1)
        dataset.create_preprocessed_dataset(mode="training", oov=False)
        dataset.create_preprocessed_dataset(mode="testing", oov=True)

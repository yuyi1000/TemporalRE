from os import listdir
from os.path import isfile, join
from nltk import tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer
from struct import unpack
from operator import itemgetter
from copy import deepcopy
from math import floor
from compares import within, greater, same_length, max_length

from sklearn.utils import shuffle

from pycorenlp import StanfordCoreNLP

import nltk
import pickle
import numpy

import itertools

import re

import xml.etree.ElementTree as ET


# print full content of numpy array
numpy.set_printoptions(threshold=numpy.nan)


class Sentence(object):
    def __init__(self, start_pos, end_pos, content):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.content = content

    def displaySentence(self):
        print ("start_pos: ", self.start_pos, ", end_pos: ", self.end_pos)
        print ("content: ", self.content)



class Entity(object):
    def __init__(self, id_, span_b, span_e, type_):
        self.id_ = id_
        self.span_b = span_b
        self.span_e = span_e
        self.type_ = type_

    def displayEntity(self):
        print ("id: ", self.id_)
        print ("span begin: ", self.span_b, ", span end: ", self.span_e)
        print ("type: ", self.type_)



class TLINK(object):
    '''
    param event_vs_event: shows if both source and target are event
    param inverse: For event_vs_event, inverse shows if source comes
    after target. for event_vs_time, inverse shows if time entity RELATIONS
    event entity.
    '''
    def __init__(self, source_b, source_e, target_b, target_e, type_, event_vs_event=True, inverse=False):
        self.source_b = source_b
        self.source_e = source_e
        self.target_b = target_b
        self.target_e = target_e
        self.type_ = type_.lower()
        self.event_vs_event = event_vs_event
        self.inverse = inverse

    def displayTLINK(self):
        print ("source begin: ", self.source_b, ", source end: ", self.source_e)
        print ("target begin: ", self.target_b, ", target end: ", self.target_e)
        print ("type: ", self.type_)
        print ("event vs event: ", event_vs_event)
        print ("inverse: ", inverse)

# TLINK2 reflects the newest structure we have been used
# We no longer use the notation of source and target
# instead, we use RELATION to show entity1 RELATION entity2
# and RELATION_inverse to show entity2 RELATION entity1
class TLINK2:
    def __init__(self, entity1_b, entity1_e, entity2_b, entity2_e, type_):
        self.entity1_b = entity1_b
        self.entity1_e = entity1_e
        self.entity2_b = entity2_b
        self.entity2_e = entity2_e

    def displayTLINK2(self):
        print ("entity1 begin: ", self.entity1_b, ", entity1 end: ", self.entity1_e)
        print ("entity2 begin: ", self.entity2_b, ", entity2 end: ", self.entity2_e)
        print ("type: ", self.type_)
        
        

################################################################
# Read word embedding from binary file
#
class WordEmbedding(object):
    def __init__(self, input_file, vocab):
        self.word_to_id = {} # word to id map
        self.id_to_word = {} # id to word map
        self.vectors = self.read_embedding(input_file, vocab)




    def add_additional_word_embedding(self, wid, vector_size, embedding_list):
        word_list = ['a', 'and', 'of', 'to', '[', ']', '(', ')', '{', '}', ',', '.', ':', '#', '|', '/', '+', '@', '$', '-']
        for word in word_list:
            self.word_to_id[word] = wid
            self.id_to_word[wid] = word
            rand_l = numpy.random.uniform(-1, 1, vector_size)
            embedding_list.append(rand_l / numpy.linalg.norm(rand_l))
            wid += 1
        return wid

        
    # read words representation from given file
    def read_embedding(self, input_file, vocabulary):
        print ("read embedding...")
        print ("input file: ", input_file)
        
        wid = 0
        em_list = []

        with open(input_file, 'rb') as f:
            cols = f.readline().strip().split()  # read first line
            vocab_size = int(cols[0].decode())   # get vocabulary size
            vector_size = int(cols[1].decode())  # get word vector size
            
            # add embedding for the padding word
            em_list.append(numpy.zeros(vector_size))
            wid += 1

            # add embedding for out of vocabulary word
            self.word_to_id['<unk>'] = wid
            self.id_to_word[wid] = '<unk>'
            em_list.append(numpy.zeros(vector_size))
            wid += 1


            # add word embedding for punctuation and stop word
            wid = self.add_additional_word_embedding(wid, vector_size, em_list)

            

            # set read format: get vector for one word in one reading operation
            fmt = str(vector_size) + 'f'

            for i in range(0, vocab_size, 1):
                # init one word with empty string
                vocab = b''

                # read char from the line till ' '
                ch = b''
                while ch != b' ':
                    vocab += ch
                    ch = f.read(1)
            
                # convert word from binary to string
                vocab = vocab.decode()

                # read one word vector
                one_vec = numpy.asarray(list(unpack(fmt, f.read(4 * vector_size))),dtype=numpy.float32)

                # if your embedding file has '\n' at the end of each line, uncomment the below line
                # if your embedding file has no '\n' at the end of each line, comment the below line
                #f.read(1)

                if vocab not in vocabulary:
                    continue

                # stored the word, word id and word representation
                self.word_to_id[vocab] = wid
                self.id_to_word[wid] = vocab
                em_list.append(one_vec)

                # increase word id
                wid += 1

        return numpy.asarray(em_list, dtype=numpy.float32)





class WordEmbedding2:

    def __init__(self, vocab):
        self.word_to_id = {} # word to id map
        self.id_to_word = {} # id to word map
        # self.vectors = self.read_embedding(input_file, vocab)
        self.vectors = self.random_embedding(vocab)
        self.positions = self.get_pos_embedding()


    def random_embedding(self, vocab):
        '''
        give each token randomized embedding
        '''
        
        wid = 0
        em_list = []

        # add embedding for the padding word
        em_list.append(numpy.zeros(300))
        wid += 1

        # add embedding for XML tags, e.g. <e>, </e>, <t>, </t>. etc
        wid = self.add_xml_tag(em_list, wid)

        for word in vocab:
            self.word_to_id[word] = wid
            self.id_to_word[wid] = word
            em_list.append(numpy.zeros(300))
            wid += 1

        # add <unk> token at the end
        self.word_to_id['<unk>'] = wid
        self.id_to_word[wid] = '<unk>'
        em_list.append(numpy.zeros(300))
            
        return numpy.asarray(em_list, dtype=numpy.float32)
        
    # add position embedding
    def get_pos_embedding(self):
        pos_embed = numpy.zeros((1, 15), dtype=numpy.int32)
        for i in range(15):
            l = numpy.zeros((1, 15), dtype=numpy.int32)
            l[0][i] = 1
            pos_embed = numpy.concatenate((pos_embed, l))
        return pos_embed
        

    # xml tags are like <e>, </e>, <t>, </t>, etc as proposed by Lin's paper
    # This function add above tags to word_to_id, id_to_word, em_list
    def add_xml_tag(self, em_list, wid):
        xml_tag_list = ["<e>", "</e>", "<t>", "</t>", "<e1>", "</e1>", "<e2>", "</e2>", "<t1>", "</t1>", "<t2>", "</t2>"]
        for xml_tag in xml_tag_list:
            self.id_to_word[wid] = xml_tag
            self.word_to_id[xml_tag] = wid
            em_list.append(numpy.zeros(300))
            wid += 1
        return wid
        
        
    # read words representation from given file
    def read_embedding(self, input_file, vocabulary):
        print ("read embedding...")
        print ("input file: ", input_file)
        
        wid = 0
        em_list = []

        # add embedding for the padding word
        em_list.append(numpy.zeros(300))
        wid += 1

        # add embedding for XML tags, e.g. <e>, </e>, <t>, </t>. etc
        wid = self.add_xml_tag(em_list, wid)
        
        with open(input_file) as f:

            for line in f:
                arr = line.split()
                word = arr[0]
                vec = arr[1:]

                # if word in self.word_to_id:
                #     print ("repeated word: ", word)
                #     print (line)
                
                if word not in vocabulary and word != '<unk>' or word in self.word_to_id or len(vec) != 300:
                    continue

                # if len(vec) != 300:
                    # print ("incorrect dimension size for the word")
                    # print ("word: ", word)
                    # print ("vec: ", vec)
                    # print ("suspicious line.")
                    # print (line)
                    # continue
                    
                self.word_to_id[word] = wid
                self.id_to_word[wid] = word
                em_list.append(vec)
                wid += 1

        # the GloVe model contains repeated words, check for that
        # ANSWER: the GloVe distinguish the difference between 'A.' and 'A. '
        # note the space
                
        # the error may come with the incorrect dimension for vec
        # i.e. some vec may not have 300 dimensions
        # check for that

                
        print ("em_list type: ", type(em_list))
        # print ("em_list length: ", len(em))
                
        return numpy.asarray(em_list, dtype=numpy.float32)



    

    
################################################################
# Read sentence from THYME data set and annotation set
#

class THYME(object):
    def __init__(self, embedding, thyme_data_path, thyme_anno_path, physio_graph_data_path='', physio_graph_anno_path='', full_dataset=False, portion=0):

        train_data_files = self.collect_id_files(thyme_data_path + '/train/')

        train_anno_files = self.collect_files(thyme_anno_path + '/train/')

        # for one_file_name in train_anno_files:
        #     one_file_name = one_file_name[:one_file_name.find('.')]
        #     open(thyme_data_path + '/train/' + one_file_name)
        # open(thyme_data_path + '/train/' + 'not_exist_file')

        # only the first file in the training data set, used for developing code.
        train_data_files_test = []
        train_data_files_test.append(train_data_files[0])
        print ("train_data_files_test length: ", len(train_data_files_test))

        # only the first file in the training anno set, used for developing code.
        train_anno_files_test = []
        train_anno_files_test.append(train_anno_files[0])
        train_anno_files_test.append(train_anno_files[1])
        train_anno_files_test.append(train_anno_files[2])
        print ("train_anno_files_test length: ", len(train_anno_files_test))
        

        dev_data_files = self.collect_id_files(thyme_data_path + '/dev/')
        dev_anno_files = self.collect_files(thyme_anno_path + '/dev/')
        dev_anno_files_test = []
        dev_anno_files_test.append(dev_anno_files[0])
        dev_anno_files_test.append(dev_anno_files[1])
        dev_anno_files_test.append(dev_anno_files[2])
        print ("dev_anno_files_test length: ", len(dev_anno_files_test))
        

        test_data_files = self.collect_id_files(thyme_data_path + '/test/')
        test_anno_files = self.collect_files(thyme_anno_path + '/test/')
        test_anno_files_test = []
        test_anno_files_test.append(test_anno_files[0])
        test_anno_files_test.append(test_anno_files[1])
        test_anno_files_test.append(test_anno_files[2])
        print ("test_anno_files_test length: ", len(test_anno_files_test))

        nlp = StanfordCoreNLP('http://localhost:9000')
        ## pattern is used to filter out unwanted sentences.
        pattern = '\[meta|\[start section|\[end section|^\n$'


        physio_graph_file = 'patient101_queries.txt'
        physio_graph_train_part_file = physio_graph_file[:physio_graph_file.rfind('.txt')] + '_' + str(portion) + '_b.txt'
        physio_graph_test_part_file = physio_graph_file[:physio_graph_file.rfind('.txt')] + '_' + str(portion) + '_a.txt'
        physio_graph_anno_file = 'patient101_queries.ann'
        physio_graph_anno_train_part_file = physio_graph_anno_file[:physio_graph_anno_file.rfind('.ann')] + '_' + str(portion) + '_b.ann'
        physio_graph_anno_test_part_file = physio_graph_anno_file[:physio_graph_anno_file.rfind('.ann')] + '_' + str(portion) + '_a.ann'
        

        print ("thyme_data_path: ", thyme_data_path)
        
        if not embedding:
            self.vocab = set()
            # if full_dataset:
            #     self.collect_vocab(pattern, nlp, thyme_data_path + '/train/', train_data_files)
            #     self.collect_vocab(pattern, nlp, thyme_data_path + '/dev/', dev_data_files)
            # else:
            #     self.collect_vocab(pattern, nlp, thyme_data_path + '/train/', train_data_files[:4])

            #     print ("train data test files")
            #     print (train_data_files[:4])
                
                # self.collect_vocab(pattern, nlp, thyme_data_path + '/dev/', dev_data_files[:3])
                
            # self.collect_vocab(pattern, nlp, thyme_data_path + '/test/', test_data_files)
            self.collect_vocab(pattern, nlp, physio_graph_data_path, [physio_graph_train_part_file])
            
            
            # print ("vocab length: ", len(self.vocab))
            # print ("vocab: ", self.vocab)
            
        else:
            self.word_to_id = embedding.word_to_id
            self.id_to_word = embedding.id_to_word
            self.vectors = embedding.vectors
            self.max_sent_len = 0
            # self.label_dict = {'NONE' : 0, 'BEFORE_LR' : 1, 'BEFORE_RL' : 2, 'CONTAINS_LR' : 3, 'CONTAINS_RL' : 4, 'OVERLAP' : 5, 'BEGINS-ON_LR' : 6,
            #                    'BEGINS-ON_RL' : 7, 'ENDS-ON_LR' : 8, 'ENDS-ON_RL' : 9}

            #################################################
            # use only three labels may improve final result
            # self.label_dict = {'NONE' : 0, 'CONTAINS_LR' : 1, 'CONTAINS_RL' : 2}
            # self.label_dict = {'none' : 0, 'contains' : 1, 'contains_1' : 2}
            self.label_dict = {'none' : 0, 'contains_lr' : 1, 'contains_rl' : 2}

            '''
            
            ###############################
            # Full dataset

            if full_dataset:
                self.train_set = self.load_data(pattern, nlp, thyme_data_path + '/train/', thyme_anno_path + '/train/', train_data_files,
                                                train_anno_files, event_vs_event=False, event_vs_time=True)
                self.train2_set = self.load_data(pattern, nlp, thyme_data_path + '/dev/', thyme_anno_path + '/dev/',
                                              dev_data_files, dev_anno_files, event_vs_event=False, event_vs_time=True)
                # self.test_set = self.load_data(pattern, nlp, thyme_data_path + '/test/', thyme_anno_path + '/test/', test_data_files,
                #                                test_anno_files, event_vs_event=False, event_vs_time=True)
                # self.closure_test_set = self.load_data(pattern, nlp, thyme_data_path + '/test/', thyme_anno_path + '/test/', test_data_files,
                #                                        test_anno_files, event_vs_event=False, event_vs_time=True, closure_test_set=True)


                # self.test_set = self.load_data(pattern, nlp, physio_graph_data_path, physio_graph_anno_path, ['patient101_queries'],
                #                                ['patient101_queries.ann'], event_vs_event=False, event_vs_time=True)

                self.dev_set = self.load_data(pattern, nlp, physio_graph_data_path, physio_graph_anno_path, [physio_graph_train_part_file],
                                               [physio_graph_anno_train_part_file], event_vs_event=False, event_vs_time=True)
                
                self.test_set = self.load_data(pattern, nlp, physio_graph_data_path, physio_graph_anno_path, [physio_graph_test_part_file],
                                               [physio_graph_anno_test_part_file], event_vs_event=False, event_vs_time=True)

                # self.closure_test_set = self.load_data(pattern, nlp, physio_graph_data_path, physio_graph_anno_path, ['patient101_queries'],
                #                                        ['patient101_queries.ann'], event_vs_event=False, event_vs_time=True, closure_test_set=True)
                self.closure_test_set = deepcopy(self.test_set)
                self.combine_two_sets(self.train_set, self.train2_set)                

                
            ################################
            # Smaller dataset for debugging
            else:
                self.train_set = self.load_data(pattern, nlp, thyme_data_path + '/train/', thyme_anno_path + '/train/',
                                                train_data_files, train_anno_files_test, event_vs_event=False, event_vs_time=True)

                self.train2_set = self.load_data(pattern, nlp, thyme_data_path + '/dev/', thyme_anno_path + '/dev/', dev_data_files,
                                              dev_anno_files_test, event_vs_event=False, event_vs_time=True)

                self.dev_set = self.load_data(pattern, nlp, physio_graph_data_path, physio_graph_anno_path, [physio_graph_train_part_file],
                                               [physio_graph_anno_train_part_file], event_vs_event=False, event_vs_time=True)

                # self.test_set = self.load_data(pattern, nlp, thyme_data_path + '/test/', thyme_anno_path + '/test/', test_data_files,
                #                                test_anno_files_test, event_vs_event=False, event_vs_time=True)
                
                self.test_set = self.load_data(pattern, nlp, physio_graph_data_path, physio_graph_anno_path, [physio_graph_test_part_file],
                                               [physio_graph_anno_test_part_file], event_vs_event=False, event_vs_time=True)

                # self.combine_two_sets(self.train_set, self.dev_set)
                self.combine_two_sets(self.train_set, self.train2_set)
                # self.closure_test_set = self.load_data(pattern, nlp, thyme_data_path + '/test/', thyme_anno_path + '/test/',
                #                                        test_data_files, test_anno_files_test, event_vs_event=False, event_vs_time=True, closure_test_set=True)
                self.closure_test_set = deepcopy(self.test_set)


            '''

            self.train_set = self.load_data(pattern, nlp, physio_graph_data_path, physio_graph_anno_path, [physio_graph_train_part_file],
                                           [physio_graph_anno_train_part_file], event_vs_event=False, event_vs_time=True)
            # dev set is not used 
            self.dev_set = self.load_data(pattern, nlp, physio_graph_data_path, physio_graph_anno_path, [physio_graph_train_part_file],
                                           [physio_graph_anno_train_part_file], event_vs_event=False, event_vs_time=True)
            self.test_set = self.load_data(pattern, nlp, physio_graph_data_path, physio_graph_anno_path, [physio_graph_test_part_file],
                                           [physio_graph_anno_test_part_file], event_vs_event=False, event_vs_time=True)
            self.closure_test_set = deepcopy(self.test_set)            
            


    def split_dataset(self, dataset, total, select):
        '''
        split dataset into two parts
        the first part starting from select / total * length(dataset), 
        ending at (select + 1) / total * length(dataset),
        the second part is comprised of the rest of dataset
        '''
        length = len(dataset[0])
        start = floor(select / total * length)
        end = floor((select + 1) / total * length)
        part1, part2 = [], []
        for i in range(len(dataset)):
            part1.append(dataset[i][start:end])
            part2.append(dataset[i][:start] + dataset[i][end:])
        return part1, part2
        

                

                
    # combine two sets into the first set
    def combine_two_sets(self, first_set, second_set):
        for i in range(len(first_set)):
            first_set[i] += second_set[i]
            first_set[i] = shuffle(first_set[i], random_state=0)
        

            

    def find_entity(self, entity_list, id_):
        for entity in entity_list:
            if entity.id_ == id_:
                return entity


    def collect_physio_graph_anno(self, path_to_file, event_vs_event, event_vs_time):
            
        event_list = []
        timex3_list = []
        entity_list = []
        tlink_list = []

        # scan for entities, events and timex3s.
        for line in open(path_to_file):
            line = line.strip()
            if line[0] == 'T':
                tid, tbody, _ = line.split('\t')
                tid = tid[1:]
                ttype, tbegin, tend = tbody.split(' ')
                if ttype == 'Timex':
                    ttype = 'TIMEX3'
                    timex3_list.append((int(tbegin), int(tend)))
                if ttype == 'Event':
                    ttype = 'EVENT'
                    event_list.append((int(tbegin), int(tend)))
                entity_list.append(Entity(int(tid), int(tbegin), int(tend), ttype))

        # for entity in entity_list:
        #     entity.displayEntity()

        # print('timex3 list length: ', len(timex3_list))
        print('timex3 list: ', timex3_list)
        print('event list: ', event_list)
        
        # scan for relations.
        for line in open(path_to_file):
            line = line.strip()
            if line[0] == 'R':
                rid, rbody = line.split('\t')
                rid = rid[1:]
                rtype, rsource, rtarget = rbody.split(' ')
                rsource = rsource[rsource.find('T'):]
                rtarget = rtarget[rtarget.find('T'):]
                esource = self.find_entity(entity_list, int(rsource[1:]))
                etarget = self.find_entity(entity_list, int(rtarget[1:]))
                source_type = esource.type_
                target_type = etarget.type_

                source_b = esource.span_b
                source_e = esource.span_e

                target_b = etarget.span_b
                target_e = etarget.span_e
                
                if event_vs_event:
                    if source_type == 'EVENT' and target_type == "EVENT":
                        tlink_list.append(TLINK(source_b, source_e, target_b, target_e, rtype, inverse=(int(source_b) > int(target_b))))
                if event_vs_time:
                    if (source_type == 'EVENT' and target_type == 'TIMEX3'):
                        tlink_list.append(TLINK(source_b, source_e, target_b, target_e, rtype, event_vs_event=False, inverse=True))
                    if (source_type == 'TIMEX3' and target_type == 'EVENT'):
                        tlink_list.append(TLINK(source_b, source_e, target_b, target_e, rtype, event_vs_event=False, inverse=False))
                        
        event_list.sort()
        timex3_list.sort()
        
        return event_list, timex3_list, tlink_list        

            
    # collect event list, timex3 list and TLINK list from annotation file
    def collect_thyme_anno(self, path_to_file, event_vs_event, event_vs_time):
        print ("collecting events...")

        tree = ET.parse(path_to_file)
        root = tree.getroot()
        annotations = root.find('annotations')

        event_list = []
        timex3_list = []
        entity_list = []
        tlink_list = []

        for entity in annotations.findall('entity'):
            span = entity.find('span').text
            # in this case, the span looks like 60,67;80,88
            # it will return 60,88
            # TODO think of a better way!
            if (span.find(';') != -1):
                span_b = span[:span.find(',')]
                span_e = span[span.rfind(',') + 1:]
                spans = [span_b, span_e]
            else:
                spans = span.split(',')

            a_type = entity.find('type').text
            id_ = entity.find('id').text
            # a common id looks like 127@e@ID001_clinic_001@gold
            # we want it return the number before the first @,
            # which is the id needed
            id_ = id_[:id_.find('@')]
            if (a_type == 'EVENT'):
                event_list.append((int(spans[0]), int(spans[1])))
            elif (a_type == 'TIMEX3'):
                timex3_list.append((int(spans[0]), int(spans[1])))
            entity_list.append(Entity(int(id_), int(spans[0]), int(spans[1]), a_type))

        for relation in annotations.findall('relation'):
            type_ = relation.find('type').text
            properties = relation.find('properties')
            if (type_ == 'TLINK'):
                source_id = properties.find('Source').text
                source_id = source_id[:source_id.find('@')]
                Type_ = properties.find('Type').text
                target_id = properties.find('Target').text
                target_id = target_id[:target_id.find('@')]


                # because the number of entity id is not continous
                # this method does not work
                # entity_s = entity_list[int(source_id) - 1]
                entity_s = self.find_entity(entity_list, int(source_id))
                source_b = getattr(entity_s, 'span_b')
                source_e = getattr(entity_s, 'span_e')
                source_type = getattr(entity_s, 'type_')

                # print ("target_id: ", target_id)
                # entity_t = entity_list[int(target_id) - 1]
                entity_t = self.find_entity(entity_list, int(target_id))
                target_b = getattr(entity_t, 'span_b')
                target_e = getattr(entity_t, 'span_e')
                target_type = getattr(entity_t, 'type_')

                if event_vs_event:
                    if source_type == 'EVENT' and target_type == "EVENT":
                        tlink_list.append(TLINK(source_b, source_e, target_b, target_e, Type_, inverse=(int(source_b) > int(target_b))))
                if event_vs_time:
                    if (source_type == 'EVENT' and target_type == 'TIMEX3'):
                        tlink_list.append(TLINK(source_b, source_e, target_b, target_e, Type_, event_vs_event=False, inverse=True))
                    if (source_type == 'TIMEX3' and target_type == 'EVENT'):
                        tlink_list.append(TLINK(source_b, source_e, target_b, target_e, Type_, event_vs_event=False, inverse=False))
                # tlink_list.append(TLINK(source_b, source_e, target_b, target_e, Type_))

        event_list.sort()
        timex3_list.sort()
        
        return event_list, timex3_list, tlink_list

    
    # we add closure only for label "contains"
    # closure means if A contains B and B contains C, then we can
    # infer A contains C and add that relation to form a closure.
    def add_closure_to_tlink_list(self, tlink_list):
        contains_tlink_list = [tlink for tlink in tlink_list if tlink.type_ == "CONTAINS"]
        added_contains_tlink_list = []
        sources = [(tlink.source_b, tlink.source_e) for tlink in contains_tlink_list]
        targets = [(tlink.target_b, tlink.target_e) for tlink in contains_tlink_list]
        for tlink in contains_tlink_list:
            source = (tlink.source_b, tlink.source_e)
            type_ = tlink.type_
            target = (tlink.target_b, tlink.target_e)
            while target in sources:
                idx = sources.index(target)
                target = targets[idx]
                if target != source:
                    added_contains_tlink_list.append(TLINK(source[0], source[1], target[0], target[1], type_))
                else:
                    break
        tlink_list.extend(added_contains_tlink_list)
        


    # input a sentence, return the character offset of each token in the sentence
    # e.g. if the input is 'hello world.' and begin_pos is 0,
    # it will return [(0, 5), (6, 11), (11, 12)]
    def token_pos(self, text, begin_pos):
        sent_tokens = []
        tokens = nltk.word_tokenize(text)
        offset = 0
        for token in tokens:
            # print ("token: ", token)

            # in the cases such as
            # [meta rev_date="02/20/2010" start_date="02/20/2010" rev="0002"]
            # word_tokenize will mistakenly take " as ''
            # this solution is not an elegent way of dealing with this problem, 
            # but it seems work fine
            if (text.find(token, offset) == -1):
                sent_tokens.append((offset + begin_pos, offset + 1 + begin_pos))
                offset += 1
            else:
                offset = text.find(token, offset)
                sent_tokens.append((offset + begin_pos, offset + len(token) + begin_pos))
                offset += len(token)
        return sent_tokens


    # same interface as token_pos function. uses Stanford NLP tokenizor as its
    # underlying parser. (outdated)
    # combine token_pos and get_sent_embed functions. uses Stanford NLP tokenizor as
    # default parser.
    # return value:
    #   sent embed: convert each word into its id by looking up table word_to_id
    def get_token_pos_and_embed(self, nlp, text, begin_pos):
        sent_tokens = []
        sent_embed = []
        output = nlp.annotate(text, properties={'annotators': 'tokenize, ssplit', 'outputFormat': 'json'})
        tokens = output['sentences'][0]['tokens']
        for token in tokens:
            # separate hyphened word. e.g. one token 'CT-scan'
            # will be split into three token 'CT', '-' and 'scan'.
            word = token['originalText']

            # if word.find('-') != -1:
            #     words = re.split('(-)', word)
            #     offset = 0
            #     for w in words:
            #         begin = begin_pos + token['characterOffsetBegin'] + offset
            #         sent_tokens.append((begin, begin + len(w)))
            #         self.append_word_to_sent_embed(sent_embed, w)
            #         offset += len(w)
            # else: 
            #     sent_tokens.append((begin_pos + token['characterOffsetBegin'], begin_pos + token['characterOffsetEnd']))
            #     self.append_word_to_sent_embed(sent_embed, word)

            sent_tokens.append((begin_pos + token['characterOffsetBegin'], begin_pos + token['characterOffsetEnd']))
            self.append_word_to_sent_embed(sent_embed, word)


                
        self.max_sent_len = max(self.max_sent_len, len(sent_embed))
        return sent_tokens, sent_embed



    def append_word_to_sent_embed(self, sent_embed, word):
        if word in self.word_to_id:
            sent_embed.append(self.word_to_id[word])
        else:
            sent_embed.append(self.word_to_id['<unk>'])        
    

    
    # sent_token_list is a list of sentence with its character offset for each token.
    # E.g. "Hello world. See you." will be represented as
    # [[(0, 5), (6, 11), (11, 12)], [(13, 16), (17, 20), (20, 21)]]
    # span_list is a list of character offsets, it can be event_list or timex3_list
    # It indicates which token in the sentence is an interested object
    # e.g. if we are interested in "world" in the above example,
    # then span_list will be [(6, 11)]
    # and it will return [[0, 1, 0], [0, 0, 0]] as the result
    def get_one_hot_encoding_for_nested_sent_list(self, sent_token_list, span_list):
        one_hot_list = []
        i = 0
        for sent_token in sent_token_list:
            one_hot = []
            j = 0
            while j < len(sent_token):
                token = sent_token[j]
                j += 1
                if (i >= len(span_list)):
                    # print ("reach the end of the span list.")
                    one_hot.append(0)
                else:
                    if (within(token, span_list[i])):
                        one_hot.append(1)
                    elif (greater(token, span_list[i])):
                        i += 1
                        j -= 1
                    else:
                        one_hot.append(0)
            one_hot_list.append(one_hot)

        return one_hot_list
    
    
    # depreciated
    # only exists as reference
    def get_sent_embed(self, text):
        # print ("----------------------------")
        # print ("sent text: ", text)
        sent_embed = []
        tokens = nltk.word_tokenize(text)

        for token in tokens:
            # print ("token: ", token, end=", ")
            if token in self.word_to_id:
                sent_embed.append(self.word_to_id[token])
            else:
                sent_embed.append(1)
        self.max_sent_len = max(self.max_sent_len, len(sent_embed))
        # print ("\n sent_embed: ", sent_embed)
        
        return sent_embed
                

    
    def get_sent_embed2(self, text, token_pos_list):
        sent_embed = []
        for b, e in token_pos_list:
            print ("b: ", b, ", e: ", e)
            token = text[b:e]
            print ("token: ", token)
            if token in self.word_to_id:
                sent_embed.append(self.word_to_id[token])
            else:
                sent_embed.append(1)
        self.max_sent_len = max(self.max_sent_len, len(sent_embed))

        return sent_embed
    
    

    # (begin, end) is the character offsets for one event or timex3
    # it will return the number of the sentence that contains these offsets
    def get_number_of_sent(self, pair, sent_token_list):
        begin = pair[0]
        end = pair[1]
        # print ("begin: ", begin, ", end: ", end)
        lo = 0
        hi = len(sent_token_list) - 1
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            sent_token = sent_token_list[mid]
            (sent_span_b, sent_span_e) = (sent_token[0][0], sent_token[len(sent_token) - 1][1])
            # print ("sent begin: ", sent_span_b, ", sent end: ", sent_span_e)
            if end < sent_span_b:
                hi = mid - 1
            elif begin > sent_span_e:
                lo = mid + 1
            else:
                # print ("in the else...")
                return mid
        return -1


    # similar to get_one_hot_encoding_for_nest_list
    # except the input flat list instead of nested list.
    def get_one_hot_encoding(self, sent_token, pair):
        begin = pair[0]
        end = pair[1]
        one_hot_vec = []
        for (token_b, token_e) in sent_token:
            if within((token_b, token_e), (begin, end)):
                one_hot_vec.append(1)
            else:
                one_hot_vec.append(0)
        return one_hot_vec
                
    

    # a typical tlink looks like
    # {source begin:  2591 , source end:  2598
    # target begin:  2678 , target end:  2693
    # type:  CONTAINS}
    # it will return a list of ((source_begin, source_end), (target_begin, target_end),
    # [one hot encoding for source], [one hot encoding for target], boolean_features, number_of_the_sentence, label)
    # boolean feature means whether the source comes before the target or not
    # if source comes before, then the boolean feature is set to be 1, if not, -1.
    def get_tlink_struct(self, sent_token_list, tlink_list):
        tlink_struct_list = []
        for tlink in tlink_list:
            tlink_min = min(tlink.source_b, tlink.source_e, tlink.target_b, tlink.target_e)
            tlink_max = max(tlink.source_b, tlink.source_e, tlink.target_b, tlink.target_e)
            sent_idx = self.get_number_of_sent((tlink_min, tlink_max), sent_token_list)
            sent_token = sent_token_list[sent_idx]
            source_one_hot = self.get_one_hot_encoding(sent_token, (tlink.source_b, tlink.source_e))
            target_one_hot = self.get_one_hot_encoding(sent_token, (tlink.target_b, tlink.target_e))
            boolean_feature = 1 if tlink.source_b < tlink.target_b else -1
            boolean_features = [boolean_feature] * len(sent_token)

            if tlink.type_ == "none":
                label = self.label_dict[tlink.type_]
            elif tlink.type_ == "contains":
                suffix = "_lr" if boolean_feature == 1 else "_rl"
                label = self.label_dict[tlink.type_ + suffix]
            else:
                label = -1
            
                

            ###########################################################
            # this is for Dligach's label notation for event-vs-time
            # if tlink.type_ == "none":
            #     label = self.label_dict[tlink.type_]
            # elif tlink.type_ == "contains":
            #     suffix = "_1" if tlink.inverse else ""
            #     label = self.label_dict[tlink.type_ + suffix]
            # else:
            #     label = -1
            
            # filter some tlink structs
            if (sent_idx != -1 and sum(source_one_hot) != 0 and sum(target_one_hot) != 0 and label != -1):
                tlink_struct_list.append(((tlink.source_b, tlink.source_e), (tlink.target_b, tlink.target_e), source_one_hot,
                                          target_one_hot, boolean_features, sent_idx, label))

                # tlink.displayTLINK()
                # print ("sent idx: ", sent_idx)
                # print ("sent token: ", sent_token_list[sent_idx])
                # print ("source one hot: ", source_one_hot)
                # print ("target one hot: ", target_one_hot)
                # print ("label: ", label)
            
        return tlink_struct_list



    def get_tlink2_struct(self, sent_token_list, tlink2_list):
        tlink2_struct_list = [];
        for tlink2 in tlink2_list:
            tlink2_min = min(tlink2.entity1_b, tlink2.entity1_e, tlink2.entity2_b, tlink2.entity2_e)
            tlink2_max = max(tlink2.entity1_b, tlink2.entity1_e, tlink2.entity2_b, tlink2.entity2_e)
            sent_idx = self.get_number_of_sent((tlink2_min, tlink2_max), sent_token_list)
            sent_token = sent_token_list[sent_idx]
            # source_one_hot = self.get_one_hot_encoding(sent_token, (tlink.source_b, tlink.source_e))
            # target_one_hot = self.get_one_hot_encoding(sent_token, (tlink.target_b, tlink.target_e))
            # boolean_feature = 1 if tlink.source_b < tlink.target_b else -1
            # boolean_features = [boolean_feature] * len(sent_token)

            label = self.label_dict[tlink2.type_]
                
            # filter some tlink2 structs
            # if (sent_idx != -1 and sum(source_one_hot) != 0 and sum(target_one_hot) != 0):
            #     tlink2_struct_list.append(((tlink2.source_b, tlink2.source_e), (tlink2.target_b, tlink2.target_e), source_one_hot,
            #                               target_one_hot, boolean_features, sent_idx, label))
            tlink2_struct_list.append((tlink2.entity1_b, tlink2.entity1_e, tlink2.entity2_b, tlink2.entity2_e), sent_idx, label)
            
        return tlink2_struct_list
        
    
        
        

    def get_entity_list_within_one_sentence(self, entity_list, sent_token_list):
        entity_one_sent_list = []
        i = 0
        for sent_token in sent_token_list:
            entity_one_sent = []
            while (i < len(entity_list)):
                entity = entity_list[i]
                # print ("i: ", i)
                sent_span = (sent_span_b, sent_span_e) = (sent_token[0][0], sent_token[len(sent_token) - 1][1])
                # print ("entity: ", entity)
                # print ("sent_span: ", sent_span)
                if within(entity, sent_span):
                    entity_one_sent.append(entity)
                    i += 1
                # entity cross the gap of two sentences
                elif sent_span_e >= entity[0] and sent_span_e < entity[1]:
                    i += 1
                # entity cross the gap of two sentences
                elif sent_span_b > entity[0] and sent_span_b <= entity[1]:
                    i += 1
                else:
                    break

            # print ("sent token: ", sent_token)

                
            entity_one_sent_list.append(entity_one_sent)
        # print ("sent token list length: ", len(sent_token_list))
        # print ("entity one sent list length: ", len(entity_one_sent_list))
        for _ in range(len(sent_token_list) - len(entity_one_sent_list)):
            entity_one_sent_list.append([])
                    
        return entity_one_sent_list


        
    
    

    # entity means either event or timex3
    # entity_pair_list looks like
    # [((272, 278), (286, 295)), ((272, 278), (315, 328)), ((272, 278), (329, 338)), ...]
    def get_entity_pair_list_within_one_sentence(self, event_list, timex3_list, sent_token_list, event_vs_event, event_vs_time):
        entity_pair_list = []

        # print ("event list: ", event_list)
        # print ("timex3 list: ", timex3_list)
        # print ("sent token list: ", sent_token_list)
        
        event_one_sent_list = self.get_entity_list_within_one_sentence(event_list, sent_token_list)
        timex3_one_sent_list = self.get_entity_list_within_one_sentence(timex3_list, sent_token_list)

        i = 0
        while i < len(sent_token_list):
            sent_token = sent_token_list[i]
            event_one_sent = event_one_sent_list[i]
            timex3_one_sent = timex3_one_sent_list[i]

            # print ("i: ", i)
            # print ("sent token: ", sent_token)
            # print ("event one sent: ", event_one_sent)
            # print ("timex3 one sent: ", timex3_one_sent)

            if event_vs_event:
                if len(event_one_sent) >= 2:
                    for pair in itertools.permutations(event_one_sent, 2):
                        entity_pair_list.append(pair)

            if event_vs_time:
                if len(event_one_sent) >= 1 and len(timex3_one_sent) >= 1:
                    for one_event in event_one_sent:
                        for one_timex3 in timex3_one_sent:
                            entity_pair_list.append((one_event, one_timex3))
                            entity_pair_list.append((one_timex3, one_event))

            i += 1

        return entity_pair_list

    
    # entity means either event or timex3
    # entity_pair_list looks like
    # [((272, 278), (286, 295)), ((272, 278), (315, 328)), ((272, 278), (329, 338)), ...]
    # compared with get_entity_pair_list_within_one_sentence, the version 2 use combination instead of permutation
    # so in version 1 both (A, B) and (B, A) will show up while in version 2 only (A, B) show up
    # Each pair reflects its position in the sentence. The first item in the pair always comes before the second in the sentence
    def get_entity_pair_list_within_one_sentence2(self, event_list, timex3_list, sent_token_list, event_vs_event, event_vs_time):
        entity_pair_list = []
        
        event_one_sent_list = self.get_entity_list_within_one_sentence(event_list, sent_token_list)
        timex3_one_sent_list = self.get_entity_list_within_one_sentence(timex3_list, sent_token_list)

        i = 0
        while i < len(sent_token_list):
            sent_token = sent_token_list[i]
            event_one_sent = event_one_sent_list[i]
            timex3_one_sent = timex3_one_sent_list[i]

            if event_vs_event:
                if len(event_one_sent) >= 2:
                    for pair in itertools.combinations(event_one_sent, 2):
                        entity_pair_list.append(pair)

            if event_vs_time:
                if len(event_one_sent) >= 1 and len(timex3_one_sent) >= 1:
                    for one_event in event_one_sent:
                        for one_timex3 in timex3_one_sent:
                            # put smaller beginning of the two into the first
                            # position of entity pair
                            if one_event[0] < one_timex3[0]:
                                entity_pair_list.append((one_event, one_timex3))
                            else:
                                entity_pair_list.append((one_timex3, one_event))

            i += 1

        return entity_pair_list
    

    # check if an entity pair is inside the tlink list
    # TODO: this is not an efficient way of checking,
    # thinking of another method.
    def inside_tlink_list(self, entity_pair, tlink_list):
        for tlink in tlink_list:
            if (entity_pair[0][0] == tlink.source_b and entity_pair[0][1] == tlink.source_e
                and entity_pair[1][0] == tlink.target_b and entity_pair[1][1] == tlink.target_e):

                return True
        return False

    
    

    # add pairs that contain no temporal relation
    def augment_tlink_list(self, entity_pair_one_sent_list, tlink_list):
        tlink_aug_list = []

        for entity_pair in entity_pair_one_sent_list:
            if not self.inside_tlink_list(entity_pair, tlink_list):
                # print ("entity pair: ", entity_pair)
                tlink_aug_list.append(TLINK(entity_pair[0][0], entity_pair[0][1], entity_pair[1][0], entity_pair[1][1], 'NONE'))

        for tlink in tlink_list:
            tlink_aug_list.append(tlink)
                
        return tlink_aug_list

    # check if an entity pair is inside the tlink list
    # The function will look at both directions. E.g. source(1, 4), target(5, 7) will be considered as the same
    # of source(5, 7), target(1, 4). Note: the format is not used correctly here.    
    # TODO: this is not an efficient way of checking,
    # thinking of another method.
    def inside_tlink_list2(self, entity_pair, tlink_list):
        for tlink in tlink_list:
            if (entity_pair[0][0] == tlink.source_b and entity_pair[0][1] == tlink.source_e
                and entity_pair[1][0] == tlink.target_b and entity_pair[1][1] == tlink.target_e) or \
                (entity_pair[1][0] == tlink.source_b and entity_pair[1][1] == tlink.source_e
                and entity_pair[0][0] == tlink.target_b and entity_pair[0][1] == tlink.target_e):
                return True
        return False
    
    
    # add pairs that contain no temporal relation
    # compared to version 1, it has a different function for judging if an pair is inside tlink list
    # The function will look at both directions. E.g. source(1, 4), target(5, 7) will be considered as the same
    # of source(5, 7), target(1, 4). Note: the format is not used correctly here.
    def augment_tlink_list2(self, entity_pair_one_sent_list, tlink_list):
        tlink_aug_list = []

        for entity_pair in entity_pair_one_sent_list:
            if not self.inside_tlink_list2(entity_pair, tlink_list):
                # print ("entity pair: ", entity_pair)
                tlink_aug_list.append(TLINK(entity_pair[0][0], entity_pair[0][1], entity_pair[1][0], entity_pair[1][1], 'NONE'))

        for tlink in tlink_list:
            tlink_aug_list.append(tlink)
                
        return tlink_aug_list


    def augment_tlink2_list(self, entity_pair_one_sent_list, tlink_list):
        tlink2_aug_list = []

        for entity_pair in entity_pair_one_sent_list:
            if not self.inside_tlink_list2(entity_pair, tlink_list):
                tlink2_aug_list.append(TLINK2(entity_pair[0][0], entity_pair[0][1], entity_pair[1][0], entity_pair[1][1], 'NONE'))

        for tlink in tlink_list:
            if tlink.source_b < tlink.target.b:
                if tlink.type_ != "NONE" and tlink.type_ != "OVERLAP":
                    tlink.type_ += "_LR"
                tlink2_aug_list.append(TLINK2(tlink.source_b, tlink.source_e, tlink.target_b, tlink.target_e, tlink.type_ + "_LR"))
            else:
                if tlink.type_ != "NONE" and tlink.type_ != "OVERLAP":
                    tlink.type_ += "_RL"
                tlink2_aug_list.append(TLINK2(tlink.target_b, tlink.target_e, tlink.source_b, tlink.source_e, tlink.type_))
        return tlink2_aug_list
        
    
    

    def list_to_array(self, one_list, max_len, aug=0):

        print ("max len: ", max_len)
        
        one_list_p = []
        for one in one_list:
            if len(one) < max_len:
                one.extend(list(numpy.zeros(max_len - len(one), dtype=numpy.int32) + aug))
            one_list_p.append(one)

        print ("one list p type: ", type(one_list_p))
        # print ("one list p: ", one_list_p)
            
        one_list_p = numpy.asarray(one_list_p, dtype=numpy.int32)
        return one_list_p


    

    
    def create_padding(self, data_set, max_sent_len):
        # sent_1v, sent_1l = self.list_to_array(data_set[0], self.max_sent_len)
        # sent_2v, sent_2l = self.list_to_array(data_set[1], self.max_sent_len)
        # data = [sent_1v, sent_1l, sent_2v, sent_2l, numpy.asarray(data_set[2], dtype=numpy.int32)]


        # print ("data_set[0]: ", data_set[0])
        # print ("data_set[1]: ", data_set[1])
        
        sent_embed_list_p = self.list_to_array(data_set[0], max_sent_len, 0)

        # pos_source_embed_list_p = self.list_to_array(data_set[1], max_sent_len)
        # pos_target_embed_list_p = self.list_to_array(data_set[2], max_sent_len)

        # pos_embed_first_entity_list_p = self.list_to_array(data_set[1], max_sent_len)
        # pos_embed_second_entity_list_p = self.list_to_array(data_set[2], max_sent_len)        
        
        # event_one_hot_list_p = self.list_to_array(data_set[3], max_sent_len)
        # timex3_one_hot_list_p = self.list_to_array(data_set[4], max_sent_len)
        # source_one_hot_list_p = self.list_to_array(data_set[5], max_sent_len)
        # target_one_hot_list_p = self.list_to_array(data_set[6], max_sent_len)

        event_one_hot_list_p = self.list_to_array(data_set[1], max_sent_len)
        timex3_one_hot_list_p = self.list_to_array(data_set[2], max_sent_len)
        source_one_hot_list_p = self.list_to_array(data_set[3], max_sent_len)
        target_one_hot_list_p = self.list_to_array(data_set[4], max_sent_len)

        
        # boolean_features_list_p = self.list_to_array(data_set[7], max_sent_len)

        # data = [sent_embed_list_p, pos_source_embed_list_p, pos_target_embed_list_p, event_one_hot_list_p, timex3_one_hot_list_p,
        #         source_one_hot_list_p, target_one_hot_list_p, boolean_features_list_p, numpy.asarray(data_set[8], dtype=numpy.int32)]

        # data = [sent_embed_list_p, pos_source_embed_list_p, pos_target_embed_list_p, event_one_hot_list_p, timex3_one_hot_list_p,
        #         source_one_hot_list_p, target_one_hot_list_p, numpy.asarray(data_set[7], dtype=numpy.int32)]

        # data = [sent_embed_list_p, pos_embed_first_entity_list_p, pos_embed_second_entity_list_p, event_one_hot_list_p, timex3_one_hot_list_p,
        #         source_one_hot_list_p, target_one_hot_list_p, numpy.asarray(data_set[7], dtype=numpy.int32)]

        data = [sent_embed_list_p, event_one_hot_list_p, timex3_one_hot_list_p, source_one_hot_list_p,
                target_one_hot_list_p, numpy.asarray(data_set[5], dtype=numpy.int32), numpy.asarray(data_set[6], dtype=numpy.int32)]

        
        # data = [sent_embed_list_p, pos_source_embed_list_p, pos_target_embed_list_p, event_one_hot_list_p, timex3_one_hot_list_p,
        #         numpy.asarray(data_set[5], dtype=numpy.int32)]
        
        # data_shuffle = shuffle(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8])
        # data_shuffle = shuffle(data[0], data[1], data[2], data[3], data[4], data[5], data[6])
        
        return data
        # return data_shuffle
        
    def create_padding_set(self):

        print ("train set length: ", len(self.train_set))
        print ("train set [0] length: ", len(self.train_set[0]))
        # print ("train set [0]: ", self.train_set[0])

        print ("train set [1] length: ", len(self.train_set[1]))
        # print ("train set [1]: ", self.train_set[1])
        print ("max length of train set [1]: ", max_length(self.train_set[1]))
        
        max_train_sent = max_length(self.train_set[0])
        max_dev_sent = max_length(self.dev_set[0])
        max_test_sent = max_length(self.test_set[0])
        max_sent_len = max(max_train_sent, max_dev_sent, max_test_sent)

        # max_sent_len = max_train_sent
        
        train_set = self.create_padding(self.train_set, max_sent_len)
        dev_set = self.create_padding(self.dev_set, max_sent_len)
        test_set = self.create_padding(self.test_set, max_sent_len)
        closure_test_set = self.create_padding(self.closure_test_set, max_sent_len)

        return train_set, dev_set, test_set, closure_test_set
        # return train_set
    

    def calculate_dataset_size(self, is_thyme_list):
        '''
        calculate the number of thyme and physiograph relations in the dataset, respectively.
        '''

        # print ("is thyme list: ")
        # print (is_thyme_list)
        
        thyme_count = numpy.count_nonzero(is_thyme_list)
        physiograph_count = len(is_thyme_list) - thyme_count
        return thyme_count, physiograph_count
    
    
        

    # calculate the total number of each label
    def calculate_label_total(self, label_list):
        label0_count = 0
        label1_count = 0
        label2_count = 0
        # label3_count = 0
        # label4_count = 0
        # label5_count = 0
        # label6_count = 0
        # label7_count = 0
        # label8_count = 0
        # label9_count = 0

        for label in label_list:
            if label == 0:
                label0_count += 1
            elif label == 1:
                label1_count += 1
            elif label == 2:
                label2_count += 1
            # elif label == 3:
            #     label3_count += 1
            # elif label == 4:
            #     label4_count += 1
            # elif label == 5:
            #     label5_count += 1                
            # elif label == 6:
            #     label6_count += 1                
            # elif label == 7:
            #     label7_count += 1                
            # elif label == 8:
            #     label8_count += 1                
            # elif label == 9:
            #     label9_count += 1                
                
        print ("number of label 0 is: ", label0_count)
        print ("number of label 1 is: ", label1_count)
        print ("number of label 2 is: ", label2_count)
        # print ("number of label 3 is: ", label3_count)
        # print ("number of label 4 is: ", label4_count)
        # print ("number of label 5 is: ", label5_count)
        # print ("number of label 6 is: ", label6_count)
        # print ("number of label 7 is: ", label7_count)
        # print ("number of label 8 is: ", label8_count)
        # print ("number of label 9 is: ", label9_count)

        # return [label0_count, label1_count, label2_count, label3_count, label4_count, label5_count, label6_count, label7_count, label8_count, label9_count]
        return [label0_count, label1_count, label2_count]

    # given an input, say [0, 0, 0, 0, 1, 0, 0]
    # return [-4, -3, -2, -1, 0, 1, 2] + 8
    def get_pos_embed(self, label_bitmap):
        pos_embed = []
        lo = label_bitmap.index(1)
        hi = lo + label_bitmap.count(1) - 1
        for i in range(len(label_bitmap)):
            if i < lo:
                pos_embed.append(max(-7, i - lo))
            elif i >= lo and i <= hi:
                pos_embed.append(0)
            else:
                pos_embed.append(min(7, i - hi))
        return [(i + 8) for i in pos_embed]
                
    def get_two_entities_pos(self, source_bitmap, target_bitmap):
        pos1_b = source_bitmap.index(1)
        pos1_e = len(source_bitmap) - 1 - source_bitmap[::-1].index(1)
        pos2_b = target_bitmap.index(1)
        pos2_e = len(target_bitmap) - 1 - target_bitmap[::-1].index(1)
        if pos2_b < pos1_b:
            pos1_b, pos1_e, pos2_b, pos2_e = pos2_b, pos2_e, pos1_b, pos1_e
        return pos1_b, pos1_e, pos2_b, pos2_e


    # add additional 0 for relevant xml tags in the bitmap (e.g. timex3_bitmap, event_bitmap)
    def get_new_bitmap_with_xml_tag(self, source_bitmap, target_bitmap, bitmap):
        new_bitmap = deepcopy(bitmap)
        pos1_b, pos1_e, pos2_b, pos2_e = self.get_two_entities_pos(source_bitmap, target_bitmap)
        new_bitmap.insert(pos1_b, 0)
        new_bitmap.insert(pos1_e + 2, 0)
        new_bitmap.insert(pos2_b + 2, 0)
        new_bitmap.insert(pos2_e + 4, 0)
        return new_bitmap
    
    # add xml tag like <e>, </e> around the source or target event and <t>, </t> around timex3.
    def get_sent_embed_with_xml_tag(self, sent_embed, source_bitmap, target_bitmap, event_bitmap, timex3_bitmap):


        new_sent_embed = deepcopy(sent_embed)
        # pos1_b = source_bitmap.index(1)
        # pos1_e = len(source_bitmap) - 1 - source_bitmap[::-1].index(1)
        # pos2_b = target_bitmap.index(1)
        # pos2_e = len(target_bitmap) - 1 - target_bitmap[::-1].index(1)
        # if pos2_b < pos1_b:
        #     pos1_b, pos1_e, pos2_b, pos2_e = pos2_b, pos2_e, pos1_b, pos1_e

        pos1_b, pos1_e, pos2_b, pos2_e = self.get_two_entities_pos(source_bitmap, target_bitmap)

        pos1 = source_bitmap.index(1)
        pos2 = target_bitmap.index(1)
        if pos2 < pos1:
            pos1, pos2 = pos2, pos1
        event_count, timex3_count = 0, 0
        for pos in [pos1, pos2]:
            if event_bitmap[pos] == 1:
                event_count += 1
            if timex3_bitmap[pos] == 1:
                timex3_count += 1
        if event_count == 1:
            if event_bitmap[pos1_b] == 1:
                new_sent_embed.insert(pos1_b, self.word_to_id["<e>"])
                new_sent_embed.insert(pos1_e + 2, self.word_to_id["</e>"])
                new_sent_embed.insert(pos2_b + 2, self.word_to_id["<t>"])
                new_sent_embed.insert(pos2_e + 4, self.word_to_id["</t>"])
            else:
                new_sent_embed.insert(pos1_b, self.word_to_id["<t>"])
                new_sent_embed.insert(pos1_e + 2, self.word_to_id["</t>"])
                new_sent_embed.insert(pos2_b + 2, self.word_to_id["<e>"])
                new_sent_embed.insert(pos2_e + 4, self.word_to_id["</e>"])                
        else:
            if event_bitmap[pos1] == 1:
                new_sent_embed.insert(pos1_b, self.word_to_id["<e1>"])
                new_sent_embed.insert(pos1_e + 2, self.word_to_id["</e1>"])
                new_sent_embed.insert(pos2_b + 2, self.word_to_id["<e2>"])
                new_sent_embed.insert(pos2_e + 4, self.word_to_id["</e2>"])
            else:
                new_sent_embed.insert(pos1_b, self.word_to_id["<t1>"])
                new_sent_embed.insert(pos1_e + 2, self.word_to_id["</t1>"])
                new_sent_embed.insert(pos2_b + 2, self.word_to_id["<t2>"])
                new_sent_embed.insert(pos2_e + 4, self.word_to_id["</t2>"])
        return new_sent_embed


    # add xml tag like <e>, </e> around the source or target event and <t>, </t> around timex3
    # and return xml tags augmented tokens
    def get_sent_token_with_xml_tag(self, sent_token, source_bitmap, target_bitmap, event_bitmap, timex3_bitmap):
        # print("timex3 bitmap", timex3_bitmap)
        # print("source bitmap: ", source_bitmap)
        # print("target bitmap: ", target_bitmap)
        new_sent_token = deepcopy(sent_token)
        pos1_b = source_bitmap.index(1)
        pos1_e = len(source_bitmap) - 1 - source_bitmap[::-1].index(1)
        pos2_b = target_bitmap.index(1)
        pos2_e = len(target_bitmap) - 1 - target_bitmap[::-1].index(1)
        if pos2_b < pos1_b:
            pos1_b, pos1_e, pos2_b, pos2_e = pos2_b, pos2_e, pos1_b, pos1_e
        event_count, timex3_count = 0, 0
        for pos in [pos1_b, pos2_b]:
            if event_bitmap[pos] == 1:
                event_count += 1
            if timex3_bitmap[pos] == 1:
                timex3_count += 1
        if event_count == 1:
            if event_bitmap[pos1_b] == 1:
                new_sent_token.insert(pos1_b, "<e>")
                new_sent_token.insert(pos1_e + 2, "</e>")
                new_sent_token.insert(pos2_b + 2, "<t>")
                new_sent_token.insert(pos2_e + 4, "</t>")
            else:
                new_sent_token.insert(pos1_b, "<t>")
                new_sent_token.insert(pos1_e + 2, "</t>")
                new_sent_token.insert(pos2_b + 2, "<e>")
                new_sent_token.insert(pos2_e + 4, "</e>")                
        else:
            if event_bitmap[pos1_b] == 1:
                new_sent_token.insert(pos1_b, "<e1>")
                new_sent_token.insert(pos1_e + 2, "</e1>")
                new_sent_token.insert(pos2_b + 2, "<e2>")
                new_sent_token.insert(pos2_e + 4, "</e2>")
            else:
                new_sent_token.insert(pos1_b, "<t1>")
                new_sent_token.insert(pos1_e + 2, "</t1>")
                new_sent_token.insert(pos2_b + 2, "<t2>")
                new_sent_token.insert(pos2_e + 4, "</t2>")
        return new_sent_token



    def get_output_buffer(self, file_content, sent_token, label):
        buffer_ = ""
        if label == 0:
            buffer_ += 'none | '
        elif label == 1:
            buffer_ += 'contains | '
        elif label == 2:
            buffer_ += 'contains_1 | '
        for token in sent_token:
            if type(token) is str:
                buffer_ += (token + ' ')
            else:
                token_b, token_e = token
                buffer_ += (file_content[token_b:token_e] + ' ')
        buffer_ += '\n'
        return buffer_


    def get_output_file_name(self, data_path, closure_test_set):
        end = data_path.rfind('/')
        begin = data_path.rfind('/', 0, end)
        file_name = data_path[begin+1:end]
        if closure_test_set:
            file_name += '_closure'
        return file_name + '.txt'


    def check_collect_anno(self, tlink_list, sent_token_list, file_content):
        result = ""
        for tlink in tlink_list:
            tlink_min = min(tlink.source_b, tlink.source_e, tlink.target_b, tlink.target_e)
            tlink_max = max(tlink.source_b, tlink.source_e, tlink.target_b, tlink.target_e)
            sent_idx = self.get_number_of_sent((tlink_min, tlink_max), sent_token_list)
            sent_token = sent_token_list[sent_idx]
            for token in sent_token:
                token_b, token_e = token
                result += file_content[token_b:token_e] + ' '
            result += '\nSource: ' + file_content[tlink.source_b:tlink.source_e]
            result += '\nTarget: ' + file_content[tlink.target_b:tlink.target_e]
            result += '\nType: ' + tlink.type_
            result += '\n------------------------------------------------------\n'
        print (result)
    
        
    # sentence pairs and their labels
    # ONLY load ONE data and anno file per time.
    # since we are only interested in annotated files
    # and each annotate file exists relevant file in data files
    # we ignore data_files for now.
    # if extract_after is set to be True, then only sentences contain "after" will be extracted
    # the argument "event_vs_time" means to extract both event vs time and time vs event
    def load_data(self, pattern, nlp, data_path, anno_path, data_files, anno_files, extract_after=False, event_vs_event=True, event_vs_time=True, closure_test_set=False):
        print ("loading data...")

        all_files_list_r = []

        # nlp = StanfordCoreNLP('http://localhost:9000')

        ## pattern is used to filter out unwanted sentences.
        # pattern = '\[meta|\[start section|\[end section|^\n$'

        output_buffer = ""
        
        for anno_file in anno_files:

            ##########################################
            # loading data from data file
            # one_file is the name of content file
            # anno_file is the name of annotation file
            one_file = anno_file[:anno_file.find('.')]

            is_thyme = False
            if anno_path.find('thyme') != -1:
                is_thyme = True

            
            print('data path: ', data_path)
            print('one file: ', one_file)
            if is_thyme:
                file_content = open(join(data_path, one_file)).read()
            else:
                file_content = open(join(data_path, one_file + '.txt')).read()
                
            # sent_token_list is a list of sentence with its character offset for each token.
            # E.g. "Hello world. See you." will be represented as
            # [[(0, 5), (6, 11), (11, 12)], [(13, 16), (17, 20), (20, 21)]]
            sent_token_list = []
            
            # sent_embed_list is a list of sentence with its embedding representation.
            # depending on the embedding model, the representation will be shown in certain way.
            # E.g. "Hello world. See you." might be represented as
            # [[122, 99, 8000], [2964, 988, 8000]]            
            sent_embed_list = []
            
            print ("one file: ", one_file)
            # print ("line of output buffer", output_buffer.count('\n'))
            
            # sent_list = self.collect_sent(data_path + one_file)
            if is_thyme:
                sent_list = self.collect_sent(nlp, pattern, data_path + one_file)
            else:
                sent_list = self.collect_sent(nlp, pattern, data_path + one_file + '.txt')                

            print ("sent list length: ", len(sent_list))

            ## haven't implemented yet. It needed only if the overhead of connecting nlp server is too large
            # sent_token_list, sent_list = self.collect_sent_and_token_pos(nlp, data_path + one_file)

            for sent in sent_list:

                # print ("sent content: ", sent.content)
                
                ## uses tokenizor from python library
                # token_pos = self.token_pos(getattr(sent, 'content'), getattr(sent, 'start_pos'))
                ## uses tokenizor from Stanford NLP package
                token_pos, sent_embed = self.get_token_pos_and_embed(nlp, sent.content, sent.start_pos)
                # sent_embed = self.get_sent_embed(getattr(sent, 'content'))
                # sent_embed = self.get_sent_embed2(sent.content, token_pos)

                # print ("token_pos: ", token_pos)
                # print ("sent embed: ", sent_embed)
                
                # print ("sent: ", sent.displaySentence())
                # print ("token_pos: ", token_pos)
                sent_token_list.append(token_pos)
                sent_embed_list.append(sent_embed)
                
            # print ("sent embed list: ", sent_embed_list)
            # print ("sent embed list length: ", len(sent_embed_list))
            # print (same_length(sent_token_list, sent_embed_list))

            
            ####################################
            # loading data from anno file

            event_token_one_hot_list = []
            timex3_token_one_hot_list = []
            print ("anno file: ", anno_file)

            
            if anno_path.find('thyme') != -1:
                event_list, timex3_list, tlink_list = self.collect_thyme_anno(anno_path + anno_file, event_vs_event, event_vs_time)
            else:
                event_list, timex3_list, tlink_list = self.collect_physio_graph_anno(anno_path + anno_file, event_vs_event, event_vs_time)
                # self.check_collect_anno(tlink_list, sent_token_list, file_content)
            
            
            if closure_test_set:
                print ("this is closure test set.")
                self.add_closure_to_tlink_list(tlink_list)

            # print ("event_list: ", event_list)
            # print ("timex3 list: ", timex3_list)

            # for tlink in tlink_list:
            #     tlink.displayTLINK()
            
            # print ("sent_token_list: ", sent_token_list)

            event_token_one_hot_list = self.get_one_hot_encoding_for_nested_sent_list(sent_token_list, event_list)
            # print ("event token list: ", event_token_one_hot_list)
            # print ("event token list length: ", len(event_token_one_hot_list))
            
            timex3_token_one_hot_list = self.get_one_hot_encoding_for_nested_sent_list(sent_token_list, timex3_list)
            # print ("timex3 token list: ", timex3_token_one_hot_list)        
            # print ("timex3 token list length: ", len(timex3_token_one_hot_list))
            # print (same_length(sent_token_list, event_token_one_hot_list))

            # tlink_struct_list = self.get_tlink_struct(sent_token_list, tlink_list)
            # print ("tlink struct list length: ", len(tlink_struct_list))

            
            entity_pair_one_sent_list = self.get_entity_pair_list_within_one_sentence2(event_list, timex3_list, sent_token_list, event_vs_event, event_vs_time)
            # print ("entity pair one sent list: ", entity_pair_one_sent_list)
            print ("entity pair one sent list length: ", len(entity_pair_one_sent_list))

            tlink_aug_list = self.augment_tlink_list2(entity_pair_one_sent_list, tlink_list)

            # use tlink2 as it reflects the newest structure
            # tlink2_aug_list = self.augment_tlink2_list(entity_pair_one_sent_list, tlink_list)

            
            # print ("tlink aug list length: ", len(tlink_aug_list))
            # for tlink in tlink_aug_list:
                # tlink.displayTLINK()

            
            
            tlink_struct_aug_list = self.get_tlink_struct(sent_token_list, tlink_aug_list)
            print ("tlink struct aug list length: ", len(tlink_struct_aug_list))


            # tlink2_struct_aug_list = self.get_tlink2_struct(sent_token_list, tlink2_aug_list)
            # print ("tlink2 struct aug list length: ", len(tlink2_struct_aug_list))
            
            
            sent_embed_list_r = []
            
            pos_embed_source_list_r = []
            pos_embed_target_list_r = []

            pos_embed_first_entity_list_r = []
            pos_embed_second_entity_list_r = []
            
            event_one_hot_list_r = []
            timex3_one_hot_list_r = []
            source_one_hot_list_r = []
            target_one_hot_list_r = []

            first_entity_bitmap_list_r = []
            second_entity_bitmap_list_r = []
            
            boolean_features_list_r = []
            label_list_r = []

            is_thyme_list_r = []

            for tlink in tlink_struct_aug_list:
                index = tlink[5]
                label_r = tlink[6]
                boolean_features = tlink[4]
                source_one_hot_r = tlink[2]
                target_one_hot_r = tlink[3]

                ###########################################
                # add first and second event position info.
                source_idx = source_one_hot_r.index(1)
                target_idx = target_one_hot_r.index(1)
                if source_idx < target_idx:
                    first_entity_bitmap = source_one_hot_r
                    second_entity_bitmap = target_one_hot_r
                else:
                    first_entity_bitmap = target_one_hot_r
                    second_entity_bitmap = source_one_hot_r
                
                pos_embed_source_r = self.get_pos_embed(source_one_hot_r)
                pos_embed_target_r = self.get_pos_embed(target_one_hot_r)

                
                ####################################################
                # add first evnet and second event in terms of position.
                # pos_embed_first_entity_r = self.get_pos_embed(first_entity_bitmap)
                # pos_embed_second_entity_r = self.get_pos_embed(second_entity_bitmap)

                
                sent_embed_r = sent_embed_list[index]
                event_one_hot_r = event_token_one_hot_list[index]
                timex3_one_hot_r = timex3_token_one_hot_list[index]

                sent_token_r = sent_token_list[index]

                # print ("index: ", index)
                # print ("label: ", label_r)
                # print ("source one hot: ", source_one_hot_r)
                # print ("target one hot: ", target_one_hot_r)
                # print ("sent embed: ", sent_embed_r)
                # print ("event one hot: ", event_one_hot_r)
                # print ("timex3 one hot: ", timex3_one_hot_r)

                
                ###################################################
                # add xml tag to sentence embedding
                xml_sent_embed_r = self.get_sent_embed_with_xml_tag(sent_embed_r, source_one_hot_r, target_one_hot_r, event_one_hot_r, timex3_one_hot_r)

                xml_event_bitmap_r = self.get_new_bitmap_with_xml_tag(source_one_hot_r, target_one_hot_r, event_one_hot_r)
                xml_timex3_bitmap_r = self.get_new_bitmap_with_xml_tag(source_one_hot_r, target_one_hot_r, timex3_one_hot_r)

                xml_first_entity_bitmap_r = self.get_new_bitmap_with_xml_tag(source_one_hot_r, target_one_hot_r, first_entity_bitmap)
                xml_second_entity_bitmap_r = self.get_new_bitmap_with_xml_tag(source_one_hot_r, target_one_hot_r, second_entity_bitmap)
                
                # add xml tag to sentence tokens
                xml_sent_token_r = self.get_sent_token_with_xml_tag(sent_token_r, source_one_hot_r, target_one_hot_r, event_one_hot_r, timex3_one_hot_r)

                output_buffer += self.get_output_buffer(file_content, xml_sent_token_r, label_r)
                

                sent_embed_list_r.append(xml_sent_embed_r)
                pos_embed_source_list_r.append(pos_embed_source_r)
                pos_embed_target_list_r.append(pos_embed_target_r)

                # pos_embed_first_entity_list_r.append(pos_embed_first_entity_r)
                # pos_embed_second_entity_list_r.append(pos_embed_second_entity_r)
                
                event_one_hot_list_r.append(xml_event_bitmap_r)
                timex3_one_hot_list_r.append(xml_timex3_bitmap_r)

                first_entity_bitmap_list_r.append(xml_first_entity_bitmap_r)
                second_entity_bitmap_list_r.append(xml_second_entity_bitmap_r)
                
                # event_one_hot_list_r.append(event_one_hot_r)
                # timex3_one_hot_list_r.append(timex3_one_hot_r)
                
                source_one_hot_list_r.append(source_one_hot_r)
                target_one_hot_list_r.append(target_one_hot_r)

                boolean_features_list_r.append(boolean_features)
                label_list_r.append(label_r)

                is_thyme_list_r.append(is_thyme)

                
            # one_file_list_r = [sent_embed_list_r, pos_embed_source_list_r, pos_embed_target_list_r, event_one_hot_list_r, timex3_one_hot_list_r,
            #                    source_one_hot_list_r, target_one_hot_list_r, boolean_features_list_r, label_list_r]

            # """
            # Although source_one_hot and target_one_hot is not used in the neural network (actual it can't), it is used in deciding the closure.
            # """
            
            # one_file_list_r = [sent_embed_list_r, pos_embed_source_list_r, pos_embed_target_list_r, event_one_hot_list_r, timex3_one_hot_list_r,
            #                    source_one_hot_list_r, target_one_hot_list_r, label_list_r]

            """
            Although source_one_hot and target_one_hot is not used in the neural network (actual it can't), it is used in deciding the closure.
            Actually, it can't even be used in deciding the closure because that will bring label knowledge to the prediction system.
            """
            
            # one_file_list_r = [sent_embed_list_r, event_one_hot_list_r, timex3_one_hot_list_r, source_one_hot_list_r, target_one_hot_list_r, label_list_r]
            
            one_file_list_r = [sent_embed_list_r, event_one_hot_list_r, timex3_one_hot_list_r,
                               first_entity_bitmap_list_r, second_entity_bitmap_list_r, label_list_r, is_thyme_list_r]
            
            # one_file_list_r = [sent_embed_list_r, pos_embed_first_entity_list_r, pos_embed_second_entity_list_r, event_one_hot_list_r, timex3_one_hot_list_r,
            #                    source_one_hot_list_r, target_one_hot_list_r, label_list_r]

            
            # one_file_list_r = [sent_embed_list_r, pos_embed_source_list_r, pos_embed_target_list_r, event_one_hot_list_r, timex3_one_hot_list_r, label_list_r]

            
            if not all_files_list_r:
                all_files_list_r = one_file_list_r
            else:
                # all_files_list_r = [all_files_list_r[0] + one_file_list_r[0], all_files_list_r[1] + one_file_list_r[1], all_files_list_r[2] + one_file_list_r[2],
                #                     all_files_list_r[3] + one_file_list_r[3], all_files_list_r[4] + one_file_list_r[4], all_files_list_r[5] + one_file_list_r[5],
                #                     all_files_list_r[6] + one_file_list_r[6], all_files_list_r[7] + one_file_list_r[7], all_files_list_r[8] + one_file_list_r[8]]

                # all_files_list_r = [all_files_list_r[0] + one_file_list_r[0], all_files_list_r[1] + one_file_list_r[1], all_files_list_r[2] + one_file_list_r[2],
                #                     all_files_list_r[3] + one_file_list_r[3], all_files_list_r[4] + one_file_list_r[4], all_files_list_r[5] + one_file_list_r[5],
                #                     all_files_list_r[6] + one_file_list_r[6], all_files_list_r[7] + one_file_list_r[7]]

                all_files_list_r = [all_files_list_r[0] + one_file_list_r[0], all_files_list_r[1] + one_file_list_r[1], all_files_list_r[2] + one_file_list_r[2],
                                    all_files_list_r[3] + one_file_list_r[3], all_files_list_r[4] + one_file_list_r[4], all_files_list_r[5] + one_file_list_r[5],
                                    all_files_list_r[6] + one_file_list_r[6]]


        output_file_name = self.get_output_file_name(data_path, closure_test_set)
        print("output file name: ", output_file_name)

        ###########################################
        # Same output format as in Dligach's paper
        # i.e. only tokens augmented with xml tags
        # e.g. She had a <e> fever </e> <t> yesterday </t> .

        # with open(output_file_name, 'w') as f:
        #     f.write(output_buffer)

        all_files_list_r_shuffle = shuffle(all_files_list_r[0], all_files_list_r[1], all_files_list_r[2],
                                           all_files_list_r[3], all_files_list_r[4], all_files_list_r[5], all_files_list_r[6])

        return all_files_list_r_shuffle



    # This collect sentence method uses function in punkt package
    # which sometimes results incorrectly separation between sentences
    # see example as shown in TODO.
            
    # def collect_sent(self, path_to_file):
    #     print ("collecting sentence...")
    #     sent_list = []
    #     # TODO: punkt sentence token mistakenly seperate phrase such as 'Dr. Crane.' as 'Dr.' and 'Crane.'
    #     f = open(path_to_file, 'r')
    #     text = f.read()
    #     for start, end in PunktSentenceTokenizer().span_tokenize(text):
    #         sent_token = text[start:end]
    #         # p_index and n_index should plus start to reflect
    #         # global character offset in the file
    #         p_index = n_index = 0
    #         while (n_index < end):
    #             p_index = sent_token.find('\n', n_index)
    #             # print ("p_index: ", p_index)
    #             if (p_index < 0):
    #                 sent_list.append(Sentence(start + n_index, end, text[start + n_index: end]))
    #                 break
    #             else:
    #                 if p_index != n_index:
    #                     sent_list.append(Sentence(start + n_index, start + p_index, text[start + n_index: start + p_index]))
    #                 n_index = p_index + 1


    #     for one_sent in sent_list:
    #         one_sent.displaySentence()
                    
    #     return sent_list



    def collect_sent(self, nlp, pattern, path_to_file):
        print ("collecting sentence...")
        sent_list = []
        f = open(path_to_file, 'r')
        # text = f.read()

        sent_offset = 0
        for line in f:
            if re.search(pattern, line) == None:
                output = nlp.annotate(line, properties={'annotators': 'tokenize, ssplit', 'outputFormat': 'json'})
                for one_sent in output['sentences']:
                    tokens = one_sent['tokens']
                    begin_idx = tokens[0]['characterOffsetBegin']
                    end_idx = tokens[-1]['characterOffsetEnd']
                    context = line[begin_idx : end_idx]
                    sent_list.append(Sentence(sent_offset + begin_idx, sent_offset + end_idx, context))
            sent_offset += len(line)
            
        # for sent in sent_list:
        #     sent.displaySentence()

        return sent_list
        

    
    def collect_id_files(self, path_to_folder):
        id_files = []
        all_files = [f for f in listdir(path_to_folder) if isfile(join(path_to_folder, f))]
        for file in all_files:
            if file[:2] == 'ID':
                id_files.append(file)
        # print ("id_files: ", id_files)
        id_files.sort()
        # print ("id_files: ", id_files)
        return id_files
            

    def collect_files(self, path_to_folder):
        files = [f for f in listdir(path_to_folder) if isfile(join(path_to_folder, f))]
        files.sort()
        # print ("files: ", files)
        # print ("files[0]: ", files[0])
        # print ("files[0] type: ", type(files[0]))
        return files

    

    # collect vocabulary of the THYME set
    def collect_vocab(self, pattern, nlp, thyme_path, file_list):
        for one_file in file_list:
            # print ("one_file: ", one_file)
            f = open(thyme_path + one_file, 'r')
            for line in f:
                if re.search(pattern, line) == None:
                    output = nlp.annotate(line, properties={'annotators': 'tokenize, ssplit', 'outputFormat': 'json'})
                    for one_sent in output['sentences']:
                        tokens = one_sent['tokens']
                        for token in tokens:
                            # separate hyphened word. e.g. one token 'CT-scan'
                            # will be split into three token 'CT', '-' and 'scan'.
                            word = token['originalText']
                            # if word.find('-') != -1:
                            #     words = re.split('(-)', word)
                            #     for w in words:
                            #         self.vocab.add(w)
                            # else:
                            #     self.vocab.add(word)    

                            self.vocab.add(word)

def main():

    # is_full_dataset = True
    is_full_dataset = False
    
    
    thyme_data_path = '/home/yuyi/corpora/THYME'
    thyme_anno_path = '/home/yuyi/corpora/THYME/thymedata-master/coloncancer'

    physio_graph_data_path = '/home/yuyi/workspace/brat-v1.3_Crunchy_Frog/data/queries/patient101/'
    physio_graph_anno_path = '/home/yuyi/workspace/brat-v1.3_Crunchy_Frog/data/queries/patient101/'

    # k = 1
    k = 5
    for i in range(k):
    
        thyme = THYME(None, thyme_data_path, thyme_anno_path, physio_graph_data_path=physio_graph_data_path, physio_graph_anno_path=physio_graph_anno_path, full_dataset=is_full_dataset, portion=i)

        # word_emb_path = '/home/yuyi/models/glove.840B.300d.txt'
        # embedding = WordEmbedding2(word_emb_path, thyme.vocab)
        # Now it just uses randomized embedding
        # embedding = WordEmbedding2(thyme.vocab)
        
        # embedding_path = '/home/yuyi/cs6890/project/data/embedding_with_xml_tag_' + str(i) + '.pkl'
        embedding_path = '/home/yuyi/cs6890/project/data/step3_embedding_with_xml_tag_' + str(i) + '.pkl'
        # embedding_path = '/home/yuyi/cs6890/project/data/embedding_test_with_xml_tag_' + str(i) + '.pkl'
        # embedding_path = '/home/yuyi/cs6890/project/data/embedding_without_xml_tag.pkl'
        # pickle.dump(embedding, open(embedding_path, 'wb'))


        embedding = pickle.load(open(embedding_path, 'rb'))

        
        
        thyme = THYME(embedding, thyme_data_path, thyme_anno_path, physio_graph_data_path=physio_graph_data_path, physio_graph_anno_path=physio_graph_anno_path, full_dataset=is_full_dataset, portion=i)
        print ("thyme: ", thyme)

        train_set, dev_set, test_set, closure_test_set = thyme.create_padding_set()

        # train_set = thyme.create_padding_set()

        print ("train set label statistics.....")
        train_label_count = thyme.calculate_label_total(train_set[-2])
        print ("dev set label statistics.....")
        dev_label_count = thyme.calculate_label_total(dev_set[-2])    
        print ("test set label statistics.....")
        test_label_count = thyme.calculate_label_total(test_set[-2])
        print ("closure test set label statistics.....")
        closure_test_label_count = thyme.calculate_label_total(closure_test_set[-2])

        train_dataset_size = thyme.calculate_dataset_size(train_set[-1])
        print ("train dataset size: ", train_dataset_size)

        test_dataset_size = thyme.calculate_dataset_size(test_set[-1])
        print ("test dataset size: ", test_dataset_size)
        

        '''
        if is_full_dataset:
            # padding_data_path = '/home/yuyi/cs6890/project/data/padding.pkl'
            # padding_data_path = '/home/yuyi/cs6890/project/data/padding_event_vs_event.pkl'
            padding_data_path = '/home/yuyi/cs6890/project/data/step2_padding_event_vs_time_with_xml_tag_' + str(i) + '.pkl'
            # padding_data_path = '/home/yuyi/cs6890/project/data/padding_event_vs_time_without_xml_tag.pkl'
            # padding_data_path = '/home/yuyi/cs6890/project/data/padding_event_vs_time_without_xml_tag_pos_embed_source.pkl'
            # pickle.dump([train_set, dev_set, test_set, closure_test_set, train_label_count], open(padding_data_path, 'wb'))
            pickle.dump([train_set, dev_set, test_set, closure_test_set, train_dataset_size], open(padding_data_path, 'wb'))

        else:

            # padding_data_test_path = '/home/yuyi/cs6890/project/data/padding_test.pkl'
            # padding_data_test_path = '/home/yuyi/cs6890/project/data/padding_test_event_vs_event.pkl'
            padding_data_test_path = '/home/yuyi/cs6890/project/data/step2_padding_test_event_vs_time_with_xml_tag_' + str(i) + '.pkl'
            # padding_data_test_path = '/home/yuyi/cs6890/project/data/padding_test_event_vs_time_without_xml_tag.pkl'
            # pickle.dump([train_set, dev_set, test_set, closure_test_set, train_label_count], open(padding_data_test_path, 'wb'))
            pickle.dump([train_set, dev_set, test_set, closure_test_set, train_dataset_size], open(padding_data_test_path, 'wb'))

        # print ("train_set[0] length: ", len(train_set[0]))
        # print ("train_set[0]: ", train_set[0])
        '''

        padding_data_path = '/home/yuyi/cs6890/project/data/step3_padding_event_vs_time_with_xml_tag_' + str(i) + '.pkl'
        pickle.dump([train_set, dev_set, test_set, closure_test_set, train_dataset_size], open(padding_data_path, 'wb'))


if __name__ == '__main__':
    main()    

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
import gzip
import pickle

import numpy
import tensorflow as tf

from train_cnn import ConvNet
# from read_file import *
from read_file import WordEmbedding2


from inspect_padding import inspect2



# Set parameters 
parser = argparse.ArgumentParser('CNN Exercise.')
parser.add_argument('--learning_rate', 
                    type=float, default=0.0001,
                    help='Initial learning rate.')
parser.add_argument('--num_epochs',
                    type=int,
                    default=300, 
                    help='Number of epochs to run trainer.')
parser.add_argument('--drop_out',
                    type=float,
                    default=0.25, 
                    help='drop out rate for last layer.')
parser.add_argument('--decay',
                    type=float,
                    default=0.001, 
                    help='Decay rate of l2 regularization.')
parser.add_argument('--batch_size', 
                    # type=int, default=50,
                    type=int, default=1, 
                    help='Batch size. Must divide evenly into the dataset sizes.')
parser.add_argument('--log_dir', 
                    type=str, 
                    default='logs', 
                    help='Directory to put logging.')
parser.add_argument('--hidden_size', 
                    type=int,
                    default='300',
                    help='.')
parser.add_argument('--model_path',
                    type=str,
                    # default='../model/cnn_event_vs_event_ckpt',
                    # default='../model/cnn_event_vs_time.ckpt',
                    default='../model/cnn_event_vs_time_200_epochs.ckpt',
                    # default='../model/cnn.ckpt',
                    help='Path of the trained model')
parser.add_argument('--embedding_path', 
                    type=str,
                    default='../data/step3_embedding_with_xml_tag.pkl',
                    # default='../data/embedding_with_xml_tag.pkl',
                    # default='../data/embedding_test_with_xml_tag.pkl',
                    help='Path of the pretrained word embedding.')
parser.add_argument('--thyme_data_dir', 
                    type=str, 
                    # default='../data/padding.pkl', 
                    # default='../data/padding_test_event_vs_event.pkl',
                    # default='../data/padding_test_event_vs_time.pkl',
                    # default='../data/padding_event_vs_event.pkl',
                    # default='../data/padding_event_vs_time_with_xml_tag.pkl',
                    # default='../data/padding_test_event_vs_time_with_xml_tag.pkl',
                    # default='../data/step2_padding_test_event_vs_time_with_xml_tag.pkl',
                    # default='../data/step2_padding_event_vs_time_with_xml_tag.pkl',
                    default='../data/step3_padding_event_vs_time_with_xml_tag.pkl',
                    help='Directory to put the thyme data.')

 
FLAGS = None
FLAGS, unparsed = parser.parse_known_args()
mode = int(sys.argv[1])


def show_sentence(sent_emb, id_to_word):
  for idx in sent_emb:
    # when idx is 0, it is a padding word.
    if idx == 0:
      break
    print (id_to_word[idx], end=" ")
  print ()



def replace_after_with_before(sent_emb):
  for i in range(len(sent_emb)):
    if sent_emb[i] == 148 or sent_emb[i] == 429:
      sent_emb[i] = 178
  

# get data set for an interested word
# E.g. we may want to investigate the word "after" and its impact
# word_to_id: {"after": 148, "After": 429, "before": 178}
def extract_word(data_set, id_to_word):
  lst = []
  sent_embed = data_set[0]
  pos_source_embed = data_set[1]
  pos_target_embed = data_set[2]
  event_one_hot = data_set[3]
  timex3_one_hot = data_set[4]
  source_one_hot = data_set[5]
  target_one_hot = data_set[6]
  boolean_features = data_set[7]
  label = data_set[8]

  n = len(data_set[0])
  print ("n: ", n)

  idx = 0

  # print ("id_to_word: ", id_to_word)

  
  for i in range(n):
    if label[i] == 1 and (148 in sent_embed[i] or 429 in sent_embed[i]):
      lst.append(i)
      print ("idx: ", idx)
      replace_after_with_before(sent_embed[i])
      show_sentence(sent_embed[i], id_to_word)
      idx += 1
      
  print ("lst: ", lst)
  print ("lst length: ", len(lst))

  filtered_data_set = [[sent_embed[i] for i in lst], [pos_source_embed[i] for i in lst], [pos_target_embed[i] for i in lst], \
                       [event_one_hot[i] for i in lst], [timex3_one_hot[i] for i in lst], [source_one_hot[i] for i in lst], \
                       [target_one_hot[i] for i in lst], [boolean_features[i] for i in lst], [label[i] for i in lst]]

  return filtered_data_set

# add validation dataset to training dataset
def combine_train_and_dev(train_set, dev_set):
  for i in range(len(train_set)):
    train_set[i] = numpy.concatenate((train_set[i], dev_set[i]))


# it creates k fold for cross validation
def create_k_fold(dataset, k):
  dataset_size = dataset[0].shape[0]

  k_fold_dev_sets = []
  k_fold_closure_sets = []
  for i in range(k):
    dev_idx_b = int(i / k * dataset_size)
    dev_idx_e = int((i + 1) / k * dataset_size)
    k_fold_closure_set = [dataset[0][dev_idx_b:dev_idx_e], dataset[1][dev_idx_b:dev_idx_e], dataset[2][dev_idx_b:dev_idx_e],
                      dataset[3][dev_idx_b:dev_idx_e], dataset[4][dev_idx_b:dev_idx_e], dataset[5][dev_idx_b:dev_idx_e]]
    k_fold_dev_set = [numpy.concatenate((dataset[0][:dev_idx_b], dataset[0][dev_idx_e:])), numpy.concatenate((dataset[1][:dev_idx_b], dataset[1][dev_idx_e:])),
                          numpy.concatenate((dataset[2][:dev_idx_b], dataset[2][dev_idx_e:])), numpy.concatenate((dataset[3][:dev_idx_b], dataset[3][dev_idx_e:])),
                          numpy.concatenate((dataset[4][:dev_idx_b], dataset[4][dev_idx_e:])), numpy.concatenate((dataset[5][:dev_idx_b], dataset[5][dev_idx_e:]))]

    k_fold_dev_sets.append(k_fold_dev_set)
    k_fold_closure_sets.append(k_fold_closure_set)
  return k_fold_dev_sets, k_fold_closure_sets
  



def main():

  class_num = 3

  total_confusion_matrix = numpy.zeros((class_num, class_num), int)
  
  # k = 5

  # for testing, set k to 1
  k = 5
  
  for i in range(k):

  
    # ======================================================================
    #  STEP 0: Load pre-trained word embeddings and the SNLI data set
    #

    embedding_path = FLAGS.embedding_path
    new_embedding_path = embedding_path[:embedding_path.rindex('.pkl')] + '_' + str(i) + '.pkl'
    
    # embedding = pickle.load(open(FLAGS.embedding_path, 'rb'))
    embedding = pickle.load(open(new_embedding_path, 'rb'))

    thyme_data_dir = FLAGS.thyme_data_dir
    new_thyme_data_dir = thyme_data_dir[:thyme_data_dir.rindex('.pkl')] + '_' + str(i) + '.pkl'
    
    # thyme = pickle.load(open(FLAGS.thyme_data_dir, 'rb'))
    thyme = pickle.load(open(new_thyme_data_dir, 'rb'))
    train_set = thyme[0]
    print("number of instances in training set: ", len(train_set[0]))
    dev_set   = thyme[1]
    # combine_train_and_dev(train_set, dev_set)
    # print("number of instances in combined training set: ", len(train_set[0]))
    test_set  = thyme[2]
    closure_test_set = thyme[3]
    # train_label_count = thyme[4]

    train_dataset_size = thyme[4]


    

    # test_after_set = extract_word(test_set, embedding.id_to_word)


    # ====================================================================
    # Use a smaller portion of training examples (e.g. ratio = 0.1) 
    # for debuging purposes.
    # Set ratio = 1 for training with all training examples.

    ratio = 1

    train_size = train_set[0].shape[0]
    idx = list(range(train_size))
    idx = numpy.asarray(idx, dtype=numpy.int32)

    # Shuffle the train set.
    for _ in range(7):
      numpy.random.shuffle(idx)

    # Get a certain ratio of the training set.
    idx = idx[0:int(idx.shape[0] * ratio)]
    sent_embed = train_set[0][idx]

    # pos_embed_source = train_set[1][idx]
    # pos_embed_target = train_set[2][idx]

    # pos_embed_first_entity = train_set[1][idx]
    # pos_embed_second_entity = train_set[2][idx]

    event_bitmap = train_set[1][idx]
    timex3_bitmap = train_set[2][idx]
    # source_bitmap = train_set[3][idx]
    # target_bitmap = train_set[4][idx]

    first_entity_bitmap = train_set[3][idx]
    second_entity_bitmap = train_set[4][idx]

    # boolean_features = train_set[7][idx]
    # label = train_set[8][idx]
    # label = train_set[7][idx]
    label = train_set[5][idx]

    # train_set = [sent_embed, pos_embed_source, pos_embed_target, event_bitmap, timex3_bitmap, source_bitmap, target_bitmap, boolean_features, label]
    # train_set = [sent_embed, pos_embed_source, pos_embed_target, event_bitmap, timex3_bitmap, source_bitmap, target_bitmap, label]
    # train_set = [sent_embed, event_bitmap, timex3_bitmap, source_bitmap, target_bitmap, label]
    # train_set = [sent_embed, event_bitmap, timex3_bitmap, first_entity_bitmap, second_entity_bitmap, label]

    # train_set = [sent_embed, pos_embed_first_entity, pos_embed_second_entity, event_bitmap, timex3_bitmap, source_bitmap, target_bitmap, label]
    # train_set = [sent_embed, pos_embed_source, pos_embed_target, event_bitmap, timex3_bitmap, label]

    # k_fold_dev_sets, k_fold_closure_test_sets = create_k_fold(closure_test_set, 10)

    # inspect2('/home/yuyi/cs6890/project/data/embedding_with_xml_tag.pkl', k_fold_dev_sets[3], entire=True)
    # inspect2('/home/yuyi/cs6890/project/data/embedding_with_xml_tag.pkl', k_fold_closure_test_sets[3], entire=True)

    # sys.exit("exit for bugging...")


    # ======================================================================
    #  STEP 1: Train a baseline model.
    #  This trains a feed forward neural network with one hidden layer.
    #
    #  Expected accuracy: 97.80%

    if mode == 1:
      cnn = ConvNet(1)
      accuracy = cnn.train_and_evaluate(FLAGS, embedding, train_set, dev_set, test_set, train_label_count)

      # Output accuracy.
      print(20 * '*' + 'model 1' + 20 * '*')
      print('accuracy is %f' % (accuracy))
      print()


    # ======================================================================
    #  STEP 2: Use one convolutional layer.
    #  
    #  Expected accuracy: 98.78%

    if mode == 2:
      cnn = ConvNet(2)

      # confusion_matrix = cnn.train_and_evaluate(FLAGS, embedding, train_set, dev_set, test_set, closure_test_set, train_label_count)


      confusion_matrix = cnn.train_and_evaluate(FLAGS, embedding, train_set, dev_set, test_set, closure_test_set, train_dataset_size)
      print(20 * '*' + 'model 2' + 20 * '*')
      print('confusion matrix: ')
      print(confusion_matrix)

      total_confusion_matrix += confusion_matrix

      # total_confusion_matrix = numpy.zeros((class_num, class_num), int)
      # for dev_set, test_set in zip(k_fold_dev_sets, k_fold_closure_test_sets):
      #   confusion_matrix = cnn.train_and_evaluate(FLAGS, embedding, train_set, dev_set, test_set, closure_test_set, train_label_count)
      #   print(20 * '*' + 'model 2' + 20 * '*')
      #   print('confusion matrix: ')
      #   print(confusion_matrix)
      #   total_confusion_matrix += confusion_matrix
      # print('total confusion matrix: ')
      # print(total_confusion_matrix)

  print('total confusion matrix: ')
  print(total_confusion_matrix)
  


if __name__ == "__main__":
  main()
  

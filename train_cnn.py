import sys
import time
import tensorflow as tf
import numpy as np

###############################################################
# Recurrent neural network class/
class ConvNet(object):
  # use model 2 as default
  def __init__(self, mode=2):
    self.mode = mode


  # Model 1: base line
  def model_1(self, rep, event_one_hot, timex3_one_hot, source_one_hot, target_one_hot, hidden_size, max_sen_len, decay):
    # print ("rep shape: ", rep.get_shape())
    # print ("event_one_hot shape: ", event_one_hot.get_shape())
    X = tf.concat([rep, tf.to_float(tf.expand_dims(event_one_hot, 2)), tf.to_float(tf.expand_dims(timex3_one_hot, 2)),
                   tf.to_float(tf.expand_dims(source_one_hot, 2)), tf.to_float(tf.expand_dims(target_one_hot, 2))], 2)
    features = tf.reshape(X, (-1, 314 * max_sen_len * 1))
    ffn = tf.layers.dense(inputs=features, units=hidden_size, activation=tf.sigmoid, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=decay))
      
    return ffn

  # Model 2: one conv layer
  def model_2(self, rep, event_one_hot, timex3_one_hot, first_entity_bitmap, second_entity_bitmap, hidden_size, max_sen_len, decay):
    # ----------------- YOUR CODE HERE ----------------------
    #

    print ("max sen length: ", max_sen_len)
    
    # X = tf.concat([rep, tf.to_float(tf.expand_dims(event_one_hot, 2)), tf.to_float(tf.expand_dims(timex3_one_hot, 2)),
    #                tf.to_float(tf.expand_dims(source_one_hot, 2)), tf.to_float(tf.expand_dims(target_one_hot, 2))], 2)
    # X = tf.concat([rep, tf.to_float(tf.expand_dims(event_one_hot, 2)), tf.to_float(tf.expand_dims(timex3_one_hot, 2))], 2)

    X = tf.concat([rep, tf.to_float(tf.expand_dims(event_one_hot, 2)), tf.to_float(tf.expand_dims(timex3_one_hot, 2)),
                   tf.to_float(tf.expand_dims(first_entity_bitmap, 2)), tf.to_float(tf.expand_dims(second_entity_bitmap, 2))], 2)
    
    print ("X shape: ", X.get_shape())
    X_4d = tf.expand_dims(X, 3)
    print ("X_4d shape: ", X_4d.get_shape())
    # print ("system exiting...")
    # sys.exit()
    # conv1 = tf.layers.conv2d(inputs=X_4d, filters=10, kernel_size=[5, 315], activation=tf.nn.relu)
    # conv1 = tf.layers.conv2d(inputs=X_4d, filters=10, kernel_size=[5, 314], activation=tf.nn.relu)

    ###########################################################
    # using 200 filters each for filter sizes 2, 3, 4 and 5
    
    conv1_size2 = tf.layers.conv2d(inputs=X_4d, filters=200, kernel_size=[2, 304], activation=tf.nn.relu)
    print ("conv1_size2 shape: ", conv1_size2.get_shape())
    # pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 1], strides=2)
    pool1_size2 = tf.layers.max_pooling2d(inputs=conv1_size2, pool_size=[max_sen_len - 1, 1], strides=[max_sen_len - 1, 1])
    print ("pool1_size2 shape: ", pool1_size2.get_shape())
    pool1_size2_flat = tf.reshape(pool1_size2, [-1, 200])
    print ("pool1_size2_flat: ", pool1_size2_flat.get_shape())

    conv1_size3 = tf.layers.conv2d(inputs=X_4d, filters=200, kernel_size=[3, 304], activation=tf.nn.relu)
    pool1_size3 = tf.layers.max_pooling2d(inputs=conv1_size3, pool_size=[max_sen_len - 2, 1], strides=[max_sen_len - 2, 1])
    pool1_size3_flat = tf.reshape(pool1_size3, [-1, 200])
    print ("pool1_size3_flat: ", pool1_size3_flat.get_shape())

    conv1_size4 = tf.layers.conv2d(inputs=X_4d, filters=200, kernel_size=[4, 304], activation=tf.nn.relu)
    pool1_size4 = tf.layers.max_pooling2d(inputs=conv1_size3, pool_size=[max_sen_len - 3, 1], strides=[max_sen_len - 3, 1])
    pool1_size4_flat = tf.reshape(pool1_size4, [-1, 200])

    conv1_size5 = tf.layers.conv2d(inputs=X_4d, filters=200, kernel_size=[5, 304], activation=tf.nn.relu)
    pool1_size5 = tf.layers.max_pooling2d(inputs=conv1_size3, pool_size=[max_sen_len - 4, 1], strides=[max_sen_len - 4, 1])
    pool1_size5_flat = tf.reshape(pool1_size5, [-1, 200])
    

    pool1_flat = tf.concat([pool1_size2_flat, pool1_size3_flat, pool1_size4_flat, pool1_size5_flat], 1)
    print("pool1_flat shape: ", pool1_flat.get_shape())
    # Do not use hidden layer in this project
    # ffn = tf.layers.dense(inputs=pool1_flat, units=hidden_size, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=decay))

    # sys.exit("exiting model 2 from ConvNet.")
    
    # return ffn

    return pool1_flat


  # get a batch for an interested word
  # E.g. we may want to investigate the word "after" and its impact
  # word_to_id: {"after": 148, "After": 429}
  # def get_word_batch(self, data_set, s, e):
  #   lst = []
  #   sent_embed = data_set[0]
  #   pos_embed_first_entity = data_set[1]
  #   pos_embed_second_entity = data_set[2]
  #   event_one_hot = data_set[3]
  #   timex3_one_hot = data_set[4]
  #   source_one_hot = data_set[5]
  #   target_one_hot = data_set[6]
  #   boolean_features = data_set[7]
  #   label = data_set[8]

  #   for i in range(s, e):
  #     if 148 in sent_embed[i] or 429 in sent_embed[i]:
  #       lst.append(i)


  #   print ("lst: ", lst)
        
  #   return [sent_embed[i] for i in lst], [pos_embed_first_entity[i] for i in lst], [pos_embed_second_entity[i] for i in lst], \
  #     [event_one_hot[i] for i in lst], [timex3_one_hot[i] for i in lst], [source_one_hot[i] for i in lst], \
  #     [target_one_hot[i] for i in lst], [boolean_features[i] for i in lst], [label[i] for i in lst]


  
  # Get a batch of data from given data set.
  def get_batch(self, data_set, s, e):

    sent_embed = data_set[0]
    # pos_embed_first_entity = data_set[1]
    # pos_embed_second_entity = data_set[2]
    event_one_hot = data_set[1]
    timex3_one_hot = data_set[2]
    # source_one_hot = data_set[3]
    # target_one_hot = data_set[4]
    first_entity_bitmap = data_set[3]
    second_entity_bitmap = data_set[4]
    
    # boolean_features = data_set[7]
    # label = data_set[8]
    label = data_set[5]
    # label = data_set[5]

    is_thyme = data_set[6]

    # position of is_thyme in the batch data
    # similar to label0_pos, label1_pos, etc.
    thyme_pos, physio_pos = [], []
    
    # positions of each label in the batch data
    # these positions are used for calculating normalized cross entropy.
    # label0_pos, label1_pos, label2_pos, label3_pos, label4_pos, label5_pos, label6_pos, label7_pos, label8_pos, label9_pos = [], [], [], [], [], [], [], [], [], []

    # for i in range(s, e):
    #   if label[i] == 0:
    #     label0_pos.append(i - s)
    #   elif label[i] == 1:
    #     label1_pos.append(i - s)
    #   elif label[i] == 2:
    #     label2_pos.append(i - s)
    #   elif label[i] == 3:
    #     label3_pos.append(i - s)
    #   elif label[i] == 4:
    #     label4_pos.append(i - s)
    #   elif label[i] == 5:
    #     label5_pos.append(i - s)
    #   elif label[i] == 6:
    #     label6_pos.append(i - s)
    #   elif label[i] == 7:
    #     label7_pos.append(i - s)
    #   elif label[i] == 8:
    #     label8_pos.append(i - s)
    #   elif label[i] == 9:
    #     label9_pos.append(i - s)

    for i in range(s, e):
      if is_thyme[i]:
        thyme_pos.append(i - s)
      else:
        physio_pos.append(i - s)
        
    
    # return sent_embed[s:e], pos_embed_first_entity[s:e], pos_embed_second_entity[s:e], event_one_hot[s:e], \
    #   timex3_one_hot[s:e], source_one_hot[s:e], target_one_hot[s:e], boolean_features[s:e], label[s:e], \
    #   label0_pos, label1_pos, label2_pos, label3_pos, label4_pos, label5_pos, label6_pos, label7_pos, label8_pos, label9_pos

    # return sent_embed[s:e], pos_embed_first_entity[s:e], pos_embed_second_entity[s:e], event_one_hot[s:e], \
    #   timex3_one_hot[s:e], source_one_hot[s:e], target_one_hot[s:e], label[s:e], \
    #   label0_pos, label1_pos, label2_pos, label3_pos, label4_pos, label5_pos, label6_pos, label7_pos, label8_pos, label9_pos

    # return sent_embed[s:e], event_one_hot[s:e], \
    #   timex3_one_hot[s:e], first_entity_bitmap[s:e], second_entity_bitmap[s:e], label[s:e], \
    #   label0_pos, label1_pos, label2_pos, label3_pos, label4_pos, label5_pos, label6_pos, label7_pos, label8_pos, label9_pos

    return sent_embed[s:e], event_one_hot[s:e], \
      timex3_one_hot[s:e], first_entity_bitmap[s:e], second_entity_bitmap[s:e], label[s:e], thyme_pos, physio_pos
  
  
    # return sent_embed[s:e], pos_embed_first_entity[s:e], pos_embed_second_entity[s:e], event_one_hot[s:e], \
    #   timex3_one_hot[s:e], label[s:e], \
    #   label0_pos, label1_pos, label2_pos, label3_pos, label4_pos, label5_pos, label6_pos, label7_pos, label8_pos, label9_pos

  

  # if event_vs_event, we just add closure for "contains" relations
  # e.g. if A contains B and B contains C, we add A contains C for closure
  # However, for event_vs_time, we must jump additional one
  # e.g. if A contains B, B contains C and C contains D, we can't draw the conclusion that
  # A contains C because that is either event vs event or time vs time
  # Therefore, only A contains D is added for the closure.
  def add_closure_to_predict_label_for_event_vs_event(self, all_sent_embed, all_source_bitmap, all_target_bitmap, all_pred_label):
    s, e, N = 0, 0, len(all_sent_embed)
    # print ("N: ", N)
    while e < N:
      s = e
      # print ("s: ", s)
      e = self.find_one_sent_range(all_sent_embed, s)
      # print ("e: ", e)
      sources = {}
      for i in range(s, e):
        pred = all_pred_label[i]
        if pred == 3 or pred == 4:
          source = all_source_bitmap[i].tolist().index(1)
          sources[source] = i
      # print ("sources: ", sources)
      for source in sources:
        visited = set([source])
        i = sources[source]
        # print ("i: ", i)
        # print (all_target_bitmap[i].tolist())
        target = all_target_bitmap[i].tolist().index(1)
        # print ("target: ", target)
        # all closure for "contains" relation
        while target in sources:
          idx = sources[target]
          # print ("idx: ", idx)
          target = all_target_bitmap[idx].tolist().index(1)
          # print ("target: ", target)
          if target not in visited:
            visited.add(target)
            all_pred_label[idx] = 3 if source < target else 4
          else:
            break

  # if event_vs_event, we just add closure for "contains" relations
  # e.g. if A contains B and B contains C, we add A contains C for closure
  # However, for event_vs_time, we must jump additional one
  # e.g. if A contains B, B contains C and C contains D, we can't draw the conclusion that
  # A contains C because that is either event vs event or time vs time
  # Therefore, only A contains D is added for the closure.
  def add_closure_to_predict_label_for_event_vs_time(self, all_sent_embed, all_source_bitmap, all_target_bitmap, all_pred_label):
    s, e, N = 0, 0, len(all_sent_embed)
    while e < N:
      s = e
      e = self.find_one_sent_range(all_sent_embed, s)
      sources = {}
      for i in range(s, e):
        pred = all_pred_label[i]
        if pred == 3 or pred == 4:
          source = all_source_bitmap[i].tolist().index(1)
          sources[source] = i
      for source in sources:
        jump_one = True
        visited = set([source])
        i = sources[source]
        target = all_target_bitmap[i].tolist().index(1)
        # all closure for "contains" relation
        while target in sources:
          idx = sources[target]
          target = all_target_bitmap[idx].tolist().index(1)
          jump_one = not jump_one
          if target not in visited and jump_one:
            visited.add(target)
            all_pred_label[idx] = 3 if source < target else 4
          else:
            break

  


          
  # assume the same sentences in the test dataset are arranged together.
  # assume the range for one sentence is dataset[s:e], it will return e.
  def find_one_sent_range(self, all_sent_embed, start_idx):
    begin_sent_embed = all_sent_embed[start_idx]
    start_idx += 1
    while start_idx < len(all_sent_embed):
      if self.is_same_sent(begin_sent_embed, all_sent_embed[start_idx]):
        # print ("start idx: ", start_idx)
        start_idx += 1
      else:
        break
    return start_idx


  def is_same_sent(self, one_sent, another_sent):
    for i, j in zip(one_sent, another_sent):
      if i != j:
        return False
    return True


  def get_confusion_matrix(self, pred_labels, true_labels, num_classes):
    confusion_matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for i in range(len(true_labels)):
      confusion_matrix[true_labels[i]][pred_labels[i]] += 1
    return np.asarray(confusion_matrix)
      
  
  
  # def train_and_evaluate(self, FLAGS, embedding, train_set, dev_set, test_set, closure_test_set, train_label_count):
  def train_and_evaluate(self, FLAGS, embedding, train_set, dev_set, test_set, closure_test_set, train_dataset_size):
    class_num     = 3
    num_epochs    = FLAGS.num_epochs
    batch_size    = FLAGS.batch_size
    learning_rate = FLAGS.learning_rate

    hidden_size = FLAGS.hidden_size
    decay = FLAGS.decay

    drop_out = FLAGS.drop_out
    model_path = FLAGS.model_path

    model_dir = model_path[:model_path.rindex('/')]

    event_vs_event = "event_vs_event" in FLAGS.thyme_data_dir
    
    # word embeding
    vectors = embedding.vectors

    # position embedding
    positions = embedding.positions

    # max length of input sequence
    max_sent_len = train_set[0].shape[1]

    # label0_count, label1_count, label2_count, label3_count, label4_count, label5_count, label6_count, label7_count, label8_count, label9_count = train_label_count
    thyme_count, physio_count = train_dataset_size

    
    print ("train_set type: ", type(train_set))
    print ("train_set length: ", len(train_set))
    
    print ("train_set[0] shape: ", train_set[0].shape)
    print ("train_set[1] shape: ", train_set[1].shape)
    print ("train_set[2] shape: ", train_set[2].shape)
    print ("train_set[3] shape: ", train_set[3].shape)
    print ("train_set[4] shape: ", train_set[4].shape)
    print ("train_set[5] shape: ", train_set[5].shape)

    # train, validate and test set size
    train_size = train_set[0].shape[0]
    
    dev_size = dev_set[0].shape[0]
    test_size = test_set[0].shape[0]
    
    print ("test size: ", test_size)
    
    with tf.Graph().as_default():
      # input data
      with tf.name_scope('input'):

        sent_embed = tf.placeholder(tf.int32, [None, max_sent_len])
        # pos_embed_first_entity = tf.placeholder(tf.int32, [None, max_sent_len])
        # pos_embed_second_entity = tf.placeholder(tf.int32, [None, max_sent_len])
        event_one_hot = tf.placeholder(tf.int32, [None, max_sent_len])
        timex3_one_hot = tf.placeholder(tf.int32, [None, max_sent_len])
        # source_one_hot = tf.placeholder(tf.int32, [None, max_sent_len])
        # target_one_hot = tf.placeholder(tf.int32, [None, max_sent_len])
        first_entity_bitmap = tf.placeholder(tf.int32, [None, max_sent_len])
        second_entity_bitmap = tf.placeholder(tf.int32, [None, max_sent_len])
        
        # boolean_features = tf.placeholder(tf.int32, [None, max_sent_len])
        is_training = tf.placeholder(tf.bool)
        
        label = tf.placeholder(tf.int32, [None]) 

        # label0_pos = tf.placeholder(tf.int32, [None])
        # label1_pos = tf.placeholder(tf.int32, [None])
        # label2_pos = tf.placeholder(tf.int32, [None])
        # label3_pos = tf.placeholder(tf.int32, [None])
        # label4_pos = tf.placeholder(tf.int32, [None])
        # label5_pos = tf.placeholder(tf.int32, [None])
        # label6_pos = tf.placeholder(tf.int32, [None])
        # label7_pos = tf.placeholder(tf.int32, [None])
        # label8_pos = tf.placeholder(tf.int32, [None])
        # label9_pos = tf.placeholder(tf.int32, [None])

        thyme_pos = tf.placeholder(tf.int32, [None])
        physio_pos = tf.placeholder(tf.int32, [None])
        
      # init embedding matrix
      # word_embedding = tf.get_variable(name="word_embedding", shape=vectors.shape, 
      #                                  initializer=tf.constant_initializer(vectors), trainable=False)

      # try randomized word embedding
      word_embedding = tf.Variable(tf.random_uniform(vectors.shape, minval=-0.05, maxval=0.05))

      
      # pos_embedding = tf.get_variable(name="pos_embedding", shape=positions.shape,
      #                                 initializer=tf.constant_initializer(positions), trainable=False)
      # embedding of out of vocabulary words
      # oov = tf.Variable(tf.random_uniform([1, vectors.shape[1]], minval=-0.05, maxval=0.05))

      # set the representation of OOV to be trainable
      # word_embedding = self.add_oov(word_embedding, vectors.shape, oov)

      # representation of the of input sentences
      # rep_1 = tf.nn.embedding_lookup(params=word_embedding, ids=sent_1) 
      # rep_2 = tf.nn.embedding_lookup(params=word_embedding, ids=sent_2)

      rep = tf.nn.embedding_lookup(params=word_embedding, ids=sent_embed)

      print ("rep shape: ", rep.get_shape())
      
      # pos_source_bitmap = tf.nn.embedding_lookup(params=pos_embedding, ids=pos_embed_first_entity)
      # pos_target_bitmap = tf.nn.embedding_lookup(params=pos_embedding, ids=pos_embed_second_entity)

      # print ("pos_source_bitmap shape: ", pos_source_bitmap.get_shape())
      
      # w = tf.Variable(tf.random_uniform([batch_size, 15, 5], minval=-0.05, maxval=0.05))
      
      # pos_source_rep = tf.matmul(pos_source_bitmap, w)
      # pos_target_rep = tf.matmul(pos_source_bitmap, w)

      # print ("pos source rep: ", pos_source_rep.get_shape())

      # rep = tf.concat([rep, pos_source_rep, pos_target_rep], axis=2)

      # print ("rep shape: ", rep.get_shape())
      
      # add boolean features, which indicate whether source comes before or after target.
      # rep = tf.concat([rep, tf.to_float(tf.expand_dims(boolean_features, 2))], 2)

      
      # model 1: base line
      if self.mode == 1:
        features = self.model_1(rep, event_one_hot, timex3_one_hot, source_one_hot, target_one_hot, hidden_size, max_sent_len, decay)
      # model 2: add one convolutional layer
      elif self.mode == 2:
        features = self.model_2(rep, event_one_hot, timex3_one_hot, first_entity_bitmap, second_entity_bitmap, hidden_size, max_sent_len, decay)

      # ======================================================================
      # define softmax layer
      # ----------------- YOUR CODE HERE ----------------------
      #

      ###################################
      # add a dropout layer
      dropout = tf.layers.dropout(inputs=features, rate=0.25, training=is_training)
      dense = tf.layers.dense(inputs=dropout, units=300, activation=tf.nn.relu)
      logits = tf.layers.dense(inputs=dense, units=class_num)
      # logits = tf.layers.dense(features, units=class_num)      
      # logits = tf.layers.dense(features, units=class_num, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=decay))

      # print ("show all trainable variables...")
      # for v in tf.trainable_variables():
      #   print (v.name)

      dense_kernel = [v for v in tf.trainable_variables() if v.name == "dense/kernel:0"][0]
      
      # ======================================================================
      # define loss function
      # ----------------- YOUR CODE HERE ----------------------
      #

      # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits) + decay * tf.nn.l2_loss(dense_kernel))
      ############################################################
      # Lin's paper does not mention the usage of regularzation
      loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits))

      
      # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits)
      # loss0 = tf.reduce_sum(tf.gather(cross_entropy, label0_pos)) / label0_count
      # loss1 = tf.reduce_sum(tf.gather(cross_entropy, label1_pos)) / label1_count
      # loss2 = tf.reduce_sum(tf.gather(cross_entropy, label2_pos)) / label2_count
      # loss3 = tf.reduce_sum(tf.gather(cross_entropy, label3_pos)) / label3_count
      # loss4 = tf.reduce_sum(tf.gather(cross_entropy, label4_pos)) / label4_count
      # loss5 = tf.reduce_sum(tf.gather(cross_entropy, label5_pos)) / label5_count
      # loss6 = tf.reduce_sum(tf.gather(cross_entropy, label6_pos)) / label6_count
      # loss7 = tf.reduce_sum(tf.gather(cross_entropy, label7_pos)) / label7_count
      # loss8 = tf.reduce_sum(tf.gather(cross_entropy, label8_pos)) / label8_count
      # loss9 = tf.reduce_sum(tf.gather(cross_entropy, label9_pos)) / label9_count

      # loss_thyme = tf.reduce_sum(tf.gather(cross_entropy, thyme_pos)) / thyme_count
      # loss_physio = tf.reduce_sum(tf.gather(cross_entropy, physio_pos)) / physio_count
      
      # loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 + loss9 + decay * tf.nn.l2_loss(dense_kernel)
      # print ("loss shape: ", loss.get_shape())

      # loss = loss_thyme + loss_physio

      # give more weight on physio graph corpus
      # loss = loss_thyme + loss_physio * 2
      
      tf.summary.scalar("loss", loss)

      # ======================================================================
      # define training operation
      #
      # train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
      # train_op = tf.train.AdadeltaOptimizer(learning_rate=1.0).minimize(loss)
      train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
      
      
      # ======================================================================
      # define accuracy operation
      # ----------------- YOUR CODE HERE ----------------------
      #
      print ("logits shape: ", logits.get_shape())
      print ("label shape: ", label.get_shape())

      pred = tf.argmax(logits, axis=1)
      print ("pred shape: ", pred.get_shape())
      confusion_matrix = tf.confusion_matrix(label, pred, num_classes=class_num)
      
      
      correct = tf.nn.in_top_k(logits, label, 1)
      print ("correct shape: ", correct.get_shape())
      correct_num = tf.reduce_sum(tf.cast(correct, tf.int32))

      saver = tf.train.Saver()
      
      # ======================================================================
      # allocate percentage of GPU memory to the session:
      # if you system does not have GPU, set has_GPU = False
      #
      has_GPU = True
      if has_GPU:
        gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        config = tf.ConfigProto(gpu_options=gpu_option)
      else:
        config = tf.ConfigProto()

      merged = tf.summary.merge_all()


      # create tensorflow session with gpu setting
      with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        tf.global_variables_initializer().run()



        
        
        #############################
        # traing the model

        epsilon = 1e-6
        repeat = 3
        prev_loss = 0
        
        for i in range(num_epochs):
          print(20 * '*', 'epoch', i+1, 20 * '*')

          start_time = time.time()
          s = 0

          # reset batch size
          batch_size = FLAGS.batch_size
          
          # total_confusion_matrix_value = np.zeros((class_num, class_num), int)
          
          while s < train_size:
            # skip the last batch for training data
            if s + batch_size >= train_size:
              break
            e = s + batch_size
            # e = min(s + batch_size, train_size)

            # (batch_sent_embed, batch_pos_source, batch_pos_target, batch_event_one_hot,
            #  batch_timex3_one_hot, batch_source_one_hot, batch_target_one_hot, batch_boolean_features, batch_label,
            #  batch_label0_pos, batch_label1_pos, batch_label2_pos, batch_label3_pos, batch_label4_pos, batch_label5_pos,
            #  batch_label6_pos, batch_label7_pos, batch_label8_pos, batch_label9_pos) \
            # = self.get_batch(train_set, s, e)

            # (batch_sent_embed, batch_pos_source, batch_pos_target, batch_event_one_hot,
            #  batch_timex3_one_hot, batch_source_one_hot, batch_target_one_hot, batch_label,
            #  batch_label0_pos, batch_label1_pos, batch_label2_pos, batch_label3_pos, batch_label4_pos, batch_label5_pos,
            #  batch_label6_pos, batch_label7_pos, batch_label8_pos, batch_label9_pos) \
            # = self.get_batch(train_set, s, e)

            # (batch_sent_embed, batch_event_one_hot,
            #  batch_timex3_one_hot, batch_first_entity_bitmap, batch_second_entity_bitmap, batch_label,
            #  batch_label0_pos, batch_label1_pos, batch_label2_pos, batch_label3_pos, batch_label4_pos, batch_label5_pos,
            #  batch_label6_pos, batch_label7_pos, batch_label8_pos, batch_label9_pos) \
            # = self.get_batch(train_set, s, e)

            
            (batch_sent_embed, batch_event_one_hot, batch_timex3_one_hot, batch_first_entity_bitmap,
             batch_second_entity_bitmap, batch_label, batch_thyme_pos, batch_physio_pos) \
            = self.get_batch(train_set, s, e)
            

            
            # (batch_sent_embed, batch_pos_source, batch_pos_target, batch_event_one_hot,
            #  batch_timex3_one_hot, batch_label,
            #  batch_label0_pos, batch_label1_pos, batch_label2_pos, batch_label3_pos, batch_label4_pos, batch_label5_pos,
            #  batch_label6_pos, batch_label7_pos, batch_label8_pos, batch_label9_pos) \
            # = self.get_batch(train_set, s, e)

            
            # print ("batch_sent_embed shape: ", batch_sent_embed.shape)
            # print ("batch_event_one_hot shape: ", batch_event_one_hot.shape)
            # print ("batch_timex3_one_hot shape: ", batch_timex3_one_hot.shape)
            # print ("batch_source_one_hot shape: ", batch_source_one_hot.shape)
            # print ("batch_target_one_hot shape: ", batch_target_one_hot.shape)
            # print ("batch_label shape: ", batch_label.shape)
            

            # print ("batch label: ", batch_label)
            
            
            _, loss_value, confusion_matrix_value \
            = sess.run([train_op, loss, confusion_matrix], \
                       feed_dict={sent_embed: batch_sent_embed,
                                  event_one_hot: batch_event_one_hot,
                                  timex3_one_hot: batch_timex3_one_hot,
                                  first_entity_bitmap: batch_first_entity_bitmap,
                                  second_entity_bitmap: batch_second_entity_bitmap,
                                  # boolean_features : batch_boolean_features,
                                  is_training: True,
                                  label: batch_label,
                                  thyme_pos: batch_thyme_pos,
                                  physio_pos: batch_physio_pos})
                                  # label1_pos: batch_label1_pos,
                                  # label2_pos: batch_label2_pos,
                                  # label3_pos: batch_label3_pos,
                                  # label4_pos: batch_label4_pos,
                                  # label5_pos: batch_label5_pos,
                                  # label6_pos: batch_label6_pos,
                                  # label7_pos: batch_label7_pos,
                                  # label8_pos: batch_label8_pos,
                                  # label9_pos: batch_label9_pos})

            # print ("s: ", s, ", e: ", e)
            # print ("loss: ", loss_value)
            # print ("confusion_matrix: ", confusion_matrix_value)
            # print ("confusion matrix type: ", type(confusion_matrix_value))
            
            # total_confusion_matrix_value += confusion_matrix_value
            
            s = e
          end_time = time.time()

          # print ("confusion matrix: ", total_confusion_matrix_value)
          
          print ('the training took: %d(s)' % (end_time - start_time), flush=True)
          print ("loss: ", loss_value)

          if abs(loss_value - prev_loss) < epsilon:
            repeat -= 1
            if repeat == 0:
              break
          else:
            repeat = 3

          prev_loss = loss_value
             

          '''
             
          ####################################################
          # evaluate model every 100 epochs on train data
          if (i + 1) % 100 == 0:

            s = 0
            total_correct = 0
            
            total_confusion_matrix_value = np.zeros((class_num, class_num), int)

            print ("train size: ", train_size)

            while s < train_size:
              # skip the last batch
              if s + batch_size >= train_size:
                break
              e = s + batch_size


              # (batch_sent_embed, batch_pos_source, batch_pos_target, batch_event_one_hot,
              #  batch_timex3_one_hot, batch_source_one_hot, batch_target_one_hot, batch_boolean_features, batch_label,
              #  batch_label0_pos, batch_label1_pos, batch_label2_pos, batch_label3_pos, batch_label4_pos, batch_label5_pos,
              #  batch_label6_pos, batch_label7_pos, batch_label8_pos, batch_label9_pos) \
              # = self.get_batch(train_set, s, e)

              # (batch_sent_embed, batch_pos_source, batch_pos_target, batch_event_one_hot,
              #  batch_timex3_one_hot, batch_source_one_hot, batch_target_one_hot, batch_label,
              #  batch_label0_pos, batch_label1_pos, batch_label2_pos, batch_label3_pos, batch_label4_pos, batch_label5_pos,
              #  batch_label6_pos, batch_label7_pos, batch_label8_pos, batch_label9_pos) \
              # = self.get_batch(train_set, s, e)

              (batch_sent_embed, batch_event_one_hot,
               batch_timex3_one_hot, batch_first_entity_bitmap, batch_second_entity_bitmap, batch_label,
               batch_label0_pos, batch_label1_pos, batch_label2_pos, batch_label3_pos, batch_label4_pos, batch_label5_pos,
               batch_label6_pos, batch_label7_pos, batch_label8_pos, batch_label9_pos) \
              = self.get_batch(train_set, s, e)

              
              # (batch_sent_embed, batch_pos_source, batch_pos_target, batch_event_one_hot,
              #  batch_timex3_one_hot, batch_label,
              #  batch_label0_pos, batch_label1_pos, batch_label2_pos, batch_label3_pos, batch_label4_pos, batch_label5_pos,
              #  batch_label6_pos, batch_label7_pos, batch_label8_pos, batch_label9_pos) \
              # = self.get_batch(train_set, s, e)

              
              pred_value, correct_num_value, confusion_matrix_value \
              = sess.run([pred, correct_num, confusion_matrix], \
                         feed_dict={sent_embed: batch_sent_embed,
                                    event_one_hot: batch_event_one_hot,
                                    timex3_one_hot: batch_timex3_one_hot,
                                    first_entity_bitmap: batch_first_entity_bitmap,
                                    second_entity_bitmap: batch_second_entity_bitmap,
                                    # boolean_features : batch_boolean_features,
                                    is_training: False,
                                    label: batch_label,
                                    label0_pos: batch_label0_pos,
                                    label1_pos: batch_label1_pos,
                                    label2_pos: batch_label2_pos,
                                    label3_pos: batch_label3_pos,
                                    label4_pos: batch_label4_pos,
                                    label5_pos: batch_label5_pos,
                                    label6_pos: batch_label6_pos,
                                    label7_pos: batch_label7_pos,
                                    label8_pos: batch_label8_pos,
                                    label9_pos: batch_label9_pos})

              total_correct += correct_num_value
              total_confusion_matrix_value += confusion_matrix_value
              s = e

            print ("confusion matrix: ")
            print (total_confusion_matrix_value)

            # print ("precision2: ", total_precision2)
            # precision2_value = total_confusion_matrix_value[2, 2] / sum(total_confusion_matrix_value[:, 2])
            # recall2_value = total_confusion_matrix_value[2, 2] / sum(total_confusion_matrix_value[2, :])
            # f1_value = 2 * precision2_value * recall2_value / (precision2_value + recall2_value)
            # print ("precision2: ", precision2_value)
            # print ("recall2: ", recall2_value)
            # print ("f1 measure: ", f1_value)

          '''
          
          #######################################################################
          # evaluate model every 100 epochs on test data, for Closure(S) ^ H
          # S is predicted value and H is manually annotated value.
          # It seems that Dligach does not use Closure(S) for calculating F measure
          # Let's skip this for now...
          # if (i + 1) % 100 == 0:

          #   print ("this is from Closure(S) ^ H ")
            
          #   all_sent_embed, all_source_bitmap, all_target_bitmap, all_pred_label, all_true_label = [], [], [], [], []
          #   s = 0
          #   total_correct = 0

          #   total_confusion_matrix_value = np.zeros((class_num, class_num), int)

          #   print ("test size: ", test_size)

          #   while s < test_size:
          #     # skip the last batch for testing data
          #     if s + batch_size >= test_size:
          #       break
          #     e = s + batch_size

          #     # (batch_sent_embed, batch_pos_source, batch_pos_target, batch_event_one_hot,
          #     #  batch_timex3_one_hot, batch_source_one_hot, batch_target_one_hot, batch_boolean_features, batch_label,
          #     #  batch_label0_pos, batch_label1_pos, batch_label2_pos, batch_label3_pos, batch_label4_pos, batch_label5_pos,
          #     #  batch_label6_pos, batch_label7_pos, batch_label8_pos, batch_label9_pos) \
          #     # = self.get_batch(test_set, s, e)              

          #     # (batch_sent_embed, batch_pos_source, batch_pos_target, batch_event_one_hot,
          #     #  batch_timex3_one_hot, batch_source_one_hot, batch_target_one_hot, batch_label,
          #     #  batch_label0_pos, batch_label1_pos, batch_label2_pos, batch_label3_pos, batch_label4_pos, batch_label5_pos,
          #     #  batch_label6_pos, batch_label7_pos, batch_label8_pos, batch_label9_pos) \
          #     # = self.get_batch(test_set, s, e)              

          #     (batch_sent_embed, batch_event_one_hot,
          #      batch_timex3_one_hot, batch_first_entity_bitmap, batch_second_entity_bitmap, batch_label,
          #      batch_label0_pos, batch_label1_pos, batch_label2_pos, batch_label3_pos, batch_label4_pos, batch_label5_pos,
          #      batch_label6_pos, batch_label7_pos, batch_label8_pos, batch_label9_pos) \
          #     = self.get_batch(test_set, s, e)              

              
          #     # (batch_sent_embed, batch_pos_source, batch_pos_target, batch_event_one_hot,
          #     #  batch_timex3_one_hot, batch_label,
          #     #  batch_label0_pos, batch_label1_pos, batch_label2_pos, batch_label3_pos, batch_label4_pos, batch_label5_pos,
          #     #  batch_label6_pos, batch_label7_pos, batch_label8_pos, batch_label9_pos) \
          #     # = self.get_batch(test_set, s, e)              

              
          #     pred_value = sess.run([pred], \
          #                           feed_dict={sent_embed: batch_sent_embed,
          #                                      event_one_hot: batch_event_one_hot,
          #                                      timex3_one_hot: batch_timex3_one_hot,
          #                                      first_entity_bitmap: batch_first_entity_bitmap,
          #                                      second_entity_bitmap: batch_second_entity_bitmap,
          #                                      # boolean_features : batch_boolean_features,
          #                                      is_training: False,
          #                                      label: batch_label,
          #                                      label0_pos: batch_label0_pos,
          #                                      label1_pos: batch_label1_pos,
          #                                      label2_pos: batch_label2_pos,
          #                                      label3_pos: batch_label3_pos,
          #                                      label4_pos: batch_label4_pos,
          #                                      label5_pos: batch_label5_pos,
          #                                      label6_pos: batch_label6_pos,
          #                                      label7_pos: batch_label7_pos,
          #                                      label8_pos: batch_label8_pos,
          #                                      label9_pos: batch_label9_pos})

              
          #     # all_sent_embed.extend(batch_sent_embed)
          #     # all_source_bitmap.extend(batch_source_one_hot)
          #     # all_target_bitmap.extend(batch_target_one_hot)
          #     # # pred value is in the form of [array([0,0,0,4,6,...])]
          #     # all_pred_label.extend(pred_value[0])
          #     # all_true_label.extend(batch_label)
              
          #     s = e

          #   # if event_vs_event:
          #   #   self.add_closure_to_predict_label_for_event_vs_event(all_sent_embed, all_source_bitmap, all_target_bitmap, all_pred_label)
          #   # else:
          #   #   self.add_closure_to_predict_label_for_event_vs_time(all_sent_embed, all_source_bitmap, all_target_bitmap, all_pred_label)
              
          #   total_confusion_matrix_value2 = self.get_confusion_matrix(all_pred_label, all_true_label, class_num)
          #   print ("confusion matrix: ")
          #   print (total_confusion_matrix_value2)
            
          #   # recall2_value = np.sum(total_confusion_matrix_value2[3:5, 3:5]) / np.sum(total_confusion_matrix_value2[3:5, :])
          #   # recall2_value = np.sum(total_confusion_matrix_value2[[3, 4], [3, 4]]) / np.sum(total_confusion_matrix_value2[3:5, :])
          #   # print ("recall2: ", recall2_value)


          # save model every 100 epochs
          # if (i + 1) % 100 == 0:
          #   saver.save(sess, model_dir + '/cnn_event_vs_time_' + str(i + 1) + '_epochs.ckpt')

          
          ###############################################################################
          # evaluate model every 100 epochs on closure test data, for S ^ Closure(H)
          # S is predicted value and H is manually annotated value.
          if (i + 1) % 100 == 0:

            print ("this is from S ^ Closure(H) ")

            # because of limited number of test case, batch size is set to be 1.
            batch_size = 1
            
            s = 0
            total_correct = 0

            total_confusion_matrix_value = np.zeros((class_num, class_num), int)

            # print ("test size: ", test_size)

            while s < test_size:
              # skip the last batch for testing data
              if s + batch_size >= test_size:
                break
              e = s + batch_size

              # (batch_sent_embed, batch_pos_source, batch_pos_target, batch_event_one_hot,
              #  batch_timex3_one_hot, batch_source_one_hot, batch_target_one_hot, batch_boolean_features, batch_label,
              #  batch_label0_pos, batch_label1_pos, batch_label2_pos, batch_label3_pos, batch_label4_pos, batch_label5_pos,
              #  batch_label6_pos, batch_label7_pos, batch_label8_pos, batch_label9_pos) \
              # = self.get_batch(closure_test_set, s, e)              

              # (batch_sent_embed, batch_pos_source, batch_pos_target, batch_event_one_hot,
              #  batch_timex3_one_hot, batch_source_one_hot, batch_target_one_hot, batch_label,
              #  batch_label0_pos, batch_label1_pos, batch_label2_pos, batch_label3_pos, batch_label4_pos, batch_label5_pos,
              #  batch_label6_pos, batch_label7_pos, batch_label8_pos, batch_label9_pos) \
              # = self.get_batch(closure_test_set, s, e)              

              # (batch_sent_embed, batch_event_one_hot,
              #  batch_timex3_one_hot, batch_first_entity_bitmap, batch_second_entity_bitmap, batch_label,
              #  batch_label0_pos, batch_label1_pos, batch_label2_pos, batch_label3_pos, batch_label4_pos, batch_label5_pos,
              #  batch_label6_pos, batch_label7_pos, batch_label8_pos, batch_label9_pos) \
              # = self.get_batch(closure_test_set, s, e)              

              (batch_sent_embed, batch_event_one_hot, batch_timex3_one_hot, batch_first_entity_bitmap,
               batch_second_entity_bitmap, batch_label, batch_thyme_pos, batch_physio_pos) \
              = self.get_batch(test_set, s, e)


              
              # (batch_sent_embed, batch_pos_source, batch_pos_target, batch_event_one_hot,
              #  batch_timex3_one_hot, batch_label,
              #  batch_label0_pos, batch_label1_pos, batch_label2_pos, batch_label3_pos, batch_label4_pos, batch_label5_pos,
              #  batch_label6_pos, batch_label7_pos, batch_label8_pos, batch_label9_pos) \
              # = self.get_batch(closure_test_set, s, e)              
              
              
              predict, correct, confusion_matrix_value \
              = sess.run([pred, correct_num, confusion_matrix], \
                         feed_dict={sent_embed: batch_sent_embed,
                                    event_one_hot: batch_event_one_hot,
                                    timex3_one_hot: batch_timex3_one_hot,
                                    first_entity_bitmap: batch_first_entity_bitmap,
                                    second_entity_bitmap: batch_second_entity_bitmap,
                                    # boolean_features : batch_boolean_features,
                                    is_training: False,
                                    label: batch_label,
                                    thyme_pos: batch_thyme_pos,
                                    physio_pos: batch_physio_pos})
                                    # label0_pos: batch_label0_pos,
                                    # label1_pos: batch_label1_pos,
                                    # label2_pos: batch_label2_pos,
                                    # label3_pos: batch_label3_pos,
                                    # label4_pos: batch_label4_pos,
                                    # label5_pos: batch_label5_pos,
                                    # label6_pos: batch_label6_pos,
                                    # label7_pos: batch_label7_pos,
                                    # label8_pos: batch_label8_pos,
                                    # label9_pos: batch_label9_pos})

              print ("predict: ", predict)
              
              total_correct += correct
              total_confusion_matrix_value += confusion_matrix_value
              s = e

            print ("confusion matrix: ")
            print (total_confusion_matrix_value)

            # precision2_value = total_confusion_matrix_value[2, 2] / sum(total_confusion_matrix_value[:, 2])
            # recall2_value = total_confusion_matrix_value[2, 2] / sum(total_confusion_matrix_value[2, :])
            # precision2_value = np.sum(total_confusion_matrix_value[3:5, 3:5]) / np.sum(total_confusion_matrix_value[:, 3:5])
            # precision2_value = np.sum(total_confusion_matrix_value[[3, 4], [3, 4]]) / np.sum(total_confusion_matrix_value[:, 3:5])
            # print ("precision2: ", precision2_value)
            
            # f1_value = 2 * precision2_value * recall2_value / (precision2_value + recall2_value)
            # print ("f1 measure: ", f1_value)     


        # save model when training is finished
        # save_path = saver.save(sess, model_path)
        # print ("Model saved in file: %s" % save_path)

        
        

        # saver.restore(sess, model_path)


        
        '''
        
        ###################################################
        # fine tuning trained model on validation dataset

        epsilon = 1e-8
        repeat = 3
        prev_loss = 0


        for i in range(num_epochs):
          print(20 * '*', 'epoch', i+1, 20 * '*')

          start_time = time.time()
          s = 0

          # reset batch size
          batch_size = 1
          
          # total_confusion_matrix_value = np.zeros((class_num, class_num), int)
          
          while s < dev_size:
            # skip the last batch for training data
            if s + batch_size >= dev_size:
              break
            e = s + batch_size
            # e = min(s + batch_size, train_size)


            # (batch_sent_embed, batch_event_one_hot,
            #  batch_timex3_one_hot, batch_first_entity_bitmap, batch_second_entity_bitmap, batch_label,
            #  batch_label0_pos, batch_label1_pos, batch_label2_pos, batch_label3_pos, batch_label4_pos, batch_label5_pos,
            #  batch_label6_pos, batch_label7_pos, batch_label8_pos, batch_label9_pos) \
            # = self.get_batch(dev_set, s, e)
            

            (batch_sent_embed, batch_event_one_hot, batch_timex3_one_hot, batch_first_entity_bitmap,
             batch_second_entity_bitmap, batch_label, batch_thyme_pos, batch_physio_pos) \
            = self.get_batch(dev_set, s, e)


            _, loss_value, confusion_matrix_value \
            = sess.run([train_op, loss, confusion_matrix], \
                       feed_dict={sent_embed: batch_sent_embed,
                                  event_one_hot: batch_event_one_hot,
                                  timex3_one_hot: batch_timex3_one_hot,
                                  first_entity_bitmap: batch_first_entity_bitmap,
                                  second_entity_bitmap: batch_second_entity_bitmap,
                                  # boolean_features : batch_boolean_features,
                                  is_training: True,
                                  label: batch_label,
                                  thyme_pos: batch_thyme_pos,
                                  physio_pos: batch_physio_pos})

            
            # _, loss_value, confusion_matrix_value \
            # = sess.run([train_op, loss, confusion_matrix], \
            #            feed_dict={sent_embed: batch_sent_embed,
            #                       event_one_hot: batch_event_one_hot,
            #                       timex3_one_hot: batch_timex3_one_hot,
            #                       first_entity_bitmap: batch_first_entity_bitmap,
            #                       second_entity_bitmap: batch_second_entity_bitmap,
            #                       # boolean_features : batch_boolean_features,
            #                       is_training: True,
            #                       label: batch_label,
            #                       label1_pos: batch_label1_pos,
            #                       label2_pos: batch_label2_pos,
            #                       label3_pos: batch_label3_pos,
            #                       label4_pos: batch_label4_pos,
            #                       label5_pos: batch_label5_pos,
            #                       label6_pos: batch_label6_pos,
            #                       label7_pos: batch_label7_pos,
            #                       label8_pos: batch_label8_pos,
            #                       label9_pos: batch_label9_pos})
            
            s = e
          end_time = time.time()

          # print ("confusion matrix: ", total_confusion_matrix_value)
          
          print ('the fine tuning took: %d(s)' % (end_time - start_time), flush=True)
          print ("loss: ", loss_value)

          if abs(loss_value - prev_loss) < epsilon:
            repeat -= 1
            if repeat == 0:
              break
          else:
            repeat = 3

          prev_loss = loss_value





          
          


          
          ###############################################################################
          # evaluate model every 100 epochs on closure test data, for S ^ Closure(H)
          # S is predicted value and H is manually annotated value.
          if (i + 1) % 100 == 0:

            print ("this is from S ^ Closure(H) ")

            # because of limited number of test case, batch size is set to be 1.
            batch_size = 1
            
            s = 0
            total_correct = 0

            total_confusion_matrix_value = np.zeros((class_num, class_num), int)

            # print ("test size: ", test_size)

            while s < test_size:
              e = s + batch_size         

              # (batch_sent_embed, batch_event_one_hot,
              #  batch_timex3_one_hot, batch_first_entity_bitmap, batch_second_entity_bitmap, batch_label,
              #  batch_label0_pos, batch_label1_pos, batch_label2_pos, batch_label3_pos, batch_label4_pos, batch_label5_pos,
              #  batch_label6_pos, batch_label7_pos, batch_label8_pos, batch_label9_pos) \
              # = self.get_batch(closure_test_set, s, e)                    
              
              (batch_sent_embed, batch_event_one_hot, batch_timex3_one_hot, batch_first_entity_bitmap,
               batch_second_entity_bitmap, batch_label, batch_thyme_pos, batch_physio_pos) \
              = self.get_batch(closure_test_set, s, e)


              _, loss_value, confusion_matrix_value \
              = sess.run([train_op, loss, confusion_matrix], \
                         feed_dict={sent_embed: batch_sent_embed,
                                    event_one_hot: batch_event_one_hot,
                                    timex3_one_hot: batch_timex3_one_hot,
                                    first_entity_bitmap: batch_first_entity_bitmap,
                                    second_entity_bitmap: batch_second_entity_bitmap,
                                    # boolean_features : batch_boolean_features,
                                    is_training: True,
                                    label: batch_label,
                                    thyme_pos: batch_thyme_pos,
                                    physio_pos: batch_physio_pos})

              
              # predict, correct, confusion_matrix_value \
              # = sess.run([pred, correct_num, confusion_matrix], \
              #            feed_dict={sent_embed: batch_sent_embed,
              #                       event_one_hot: batch_event_one_hot,
              #                       timex3_one_hot: batch_timex3_one_hot,
              #                       first_entity_bitmap: batch_first_entity_bitmap,
              #                       second_entity_bitmap: batch_second_entity_bitmap,
              #                       # boolean_features : batch_boolean_features,
              #                       is_training: False,
              #                       label: batch_label,
              #                       label0_pos: batch_label0_pos,
              #                       label1_pos: batch_label1_pos,
              #                       label2_pos: batch_label2_pos,
              #                       label3_pos: batch_label3_pos,
              #                       label4_pos: batch_label4_pos,
              #                       label5_pos: batch_label5_pos,
              #                       label6_pos: batch_label6_pos,
              #                       label7_pos: batch_label7_pos,
              #                       label8_pos: batch_label8_pos,
              #                       label9_pos: batch_label9_pos})

              print ("predict: ", predict)
              
              total_correct += correct
              total_confusion_matrix_value += confusion_matrix_value
              s = e

            print ("confusion matrix: ")
            print (total_confusion_matrix_value)


        '''




        
        #########################################
        # evaluate the trained model on test set

        # saver.restore(sess, model_path)
        # print ("Model restored from: %s" % model_path)


        s = 0
        total_correct = 0

        total_confusion_matrix_value = np.zeros((class_num, class_num), int)

        # because of limited number of test case, batch size is set to be 1.
        batch_size = 1
        
        print ("test size: ", test_size)
        
        while s < test_size:
          
          e = s + batch_size


          # (batch_sent_embed, batch_pos_source, batch_pos_target, batch_event_one_hot, \
          #  batch_timex3_one_hot, batch_source_one_hot, batch_target_one_hot, batch_boolean_features, batch_label, \
          #  batch_label0_pos, batch_label1_pos, batch_label2_pos, batch_label3_pos, batch_label4_pos, batch_label5_pos,
          #  batch_label6_pos, batch_label7_pos, batch_label8_pos, batch_label9_pos) \
          # = self.get_batch(test_set, s, e)          

          # (batch_sent_embed, batch_pos_source, batch_pos_target, batch_event_one_hot, \
          #  batch_timex3_one_hot, batch_source_one_hot, batch_target_one_hot, batch_label, \
          #  batch_label0_pos, batch_label1_pos, batch_label2_pos, batch_label3_pos, batch_label4_pos, batch_label5_pos,
          #  batch_label6_pos, batch_label7_pos, batch_label8_pos, batch_label9_pos) \
          # = self.get_batch(test_set, s, e)          

          # (batch_sent_embed, batch_event_one_hot, \
          #  batch_timex3_one_hot, batch_first_entity_bitmap, batch_second_entity_bitmap, batch_label, \
          #  batch_label0_pos, batch_label1_pos, batch_label2_pos, batch_label3_pos, batch_label4_pos, batch_label5_pos,
          #  batch_label6_pos, batch_label7_pos, batch_label8_pos, batch_label9_pos) \
          # = self.get_batch(test_set, s, e)          

          (batch_sent_embed, batch_event_one_hot, \
           batch_timex3_one_hot, batch_first_entity_bitmap, batch_second_entity_bitmap, batch_label, batch_thyme_pos, batch_physio_pos) \
          = self.get_batch(test_set, s, e)          

             
          
          # (batch_sent_embed, batch_pos_source, batch_pos_target, batch_event_one_hot, \
          #  batch_timex3_one_hot, batch_label, \
          #  batch_label0_pos, batch_label1_pos, batch_label2_pos, batch_label3_pos, batch_label4_pos, batch_label5_pos,
          #  batch_label6_pos, batch_label7_pos, batch_label8_pos, batch_label9_pos) \
          # = self.get_batch(test_set, s, e)          

          
          correct, confusion_matrix_value, pred_value \
          = sess.run([correct_num, confusion_matrix, pred], \
                     feed_dict={sent_embed: batch_sent_embed,
                                event_one_hot: batch_event_one_hot,
                                timex3_one_hot: batch_timex3_one_hot,
                                first_entity_bitmap: batch_first_entity_bitmap,
                                second_entity_bitmap: batch_second_entity_bitmap,
                                # boolean_features : batch_boolean_features,
                                is_training: False,
                                label: batch_label,
                                thyme_pos: batch_thyme_pos,
                                physio_pos: batch_physio_pos})
                                # label0_pos: batch_label0_pos,
                                # label1_pos: batch_label1_pos,
                                # label2_pos: batch_label2_pos,
                                # label3_pos: batch_label3_pos,
                                # label4_pos: batch_label4_pos,
                                # label5_pos: batch_label5_pos,
                                # label6_pos: batch_label6_pos,
                                # label7_pos: batch_label7_pos,
                                # label8_pos: batch_label8_pos,
                                # label9_pos: batch_label9_pos})
          


          # (batch_sent_embed, batch_pos_source, batch_pos_target, batch_event_one_hot, \
          #  batch_timex3_one_hot, batch_source_one_hot, batch_target_one_hot, batch_boolean_features, batch_label) \
          #  = self.get_word_batch(train_set, s, e)

          # correct, confusion_matrix_value \
          # = sess.run([accuracy, confusion_matrix], \
          #            feed_dict={sent_embed: batch_sent_embed,
          #                       pos_embed_first_entity: batch_pos_source,
          #                       pos_embed_second_entity: batch_pos_target,
          #                       event_one_hot: batch_event_one_hot,
          #                       timex3_one_hot: batch_timex3_one_hot,
          #                       source_one_hot: batch_source_one_hot,
          #                       target_one_hot: batch_target_one_hot,
          #                       boolean_features : batch_boolean_features,
          #                       label: batch_label})
          

          print ("s: ", s, ", e: ", e)
          print ("pred: ", pred_value)
          
          total_correct += correct
          total_confusion_matrix_value += confusion_matrix_value
          s = e


          
        print ("confusion matrix: ")
        print (total_confusion_matrix_value)

        # precision2_value = total_confusion_matrix_value[2, 2] / sum(total_confusion_matrix_value[:, 2])
        # recall2_value = total_confusion_matrix_value[2, 2] / sum(total_confusion_matrix_value[2, :])
        # precision2_value = np.sum(total_confusion_matrix_value[3:5, 3:5]) / np.sum(total_confusion_matrix_value[:, 3:5])
        # recall2_value = np.sum(total_confusion_matrix_value[3:5, 3:5]) / np.sum(total_confusion_matrix_value[3:5, :])
        # f1_value = 2 * precision2_value * recall2_value / (precision2_value + recall2_value)
        # print ("precision2: ", precision2_value)
        # print ("recall2: ", recall2_value)
        # print ("f1 measure: ", f1_value)
        

        # return total_correct / test_size
        return total_confusion_matrix_value

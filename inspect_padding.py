

import pickle
from read_file import WordEmbedding2


def inspect2(embedding_file,  data, entire=False):
    
    embed = pickle.load(open(embedding_file, 'rb'))
    id_to_word = embed.id_to_word
    word_to_id = embed.word_to_id
    vectors = embed.vectors


    if entire:
        id_list = [i for i in range(len(data[0]))]
    else:
        # id_list = [i for i in range(3)]
        id_list = [0, 1, 2, 3, 4, 5, 10, 15, 22, 100, 137, 159, 1000, 1003]

    for i in id_list:

        list_idx = i
        
        sent_embed = data[0][i]
        # pos_embed_source = data[1][i]
        # pos_embed_target = data[2][i]
        # event_bitmap = data[3][i]
        # timex3_bitmap = data[4][i]
        # source_bitmap = data[5][i]
        # target_bitmap = data[6][i]
        # boolean_features = data[7][i]
        # label = data[8][i]

        event_bitmap = data[1][i]
        timex3_bitmap = data[2][i]
        # source_bitmap = data[3][i]
        # target_bitmap = data[4][i]

        first_entity_bitmap = data[3][i]
        second_entity_bitmap = data[4][i]
        
        label = data[5][i]

        is_thyme = data[6][i]
        
        sentence = ''
        event = ''
        timex3 = ''
        # source = ''
        # target = ''

        first_entity = ''
        second_entity = ''
        
        for idx in sent_embed:
            if idx == 0:
                break
            sentence += id_to_word[idx] + ' '

        for i, b in enumerate(event_bitmap):
            if b == 1:
                event += id_to_word[sent_embed[i]] + ' '

        for i, b in enumerate(timex3_bitmap):
            if b == 1:
                timex3 += id_to_word[sent_embed[i]] + ' '


        # for i, b in enumerate(source_bitmap):
        #     if b == 1:
        #         source += id_to_word[sent_embed[i]] + ' '

        for i, b in enumerate(first_entity_bitmap):
            if b == 1:
                first_entity += id_to_word[sent_embed[i]] + ' '

                
        # for i, b in enumerate(target_bitmap):
        #     if b == 1:
        #         target += id_to_word[sent_embed[i]] + ' '

        for i, b in enumerate(second_entity_bitmap):
            if b == 1:
                second_entity += id_to_word[sent_embed[i]] + ' '

        print ("list idx: ", list_idx + 1)
                
        print ("sentence: ", sentence)
        print ("event: ", event)
        print ("timex3: ", timex3)
        # print ("source: ", source)
        # print ("target: ", target)

        print ("first entity: ", first_entity)
        print ("second entity: ", second_entity)
        
        print ("label: ", label)
        print ("is thyme: ", is_thyme)
        
        # print ("raw event bitmap: ", event_bitmap)
        # print ("raw timex3 bitmap: ", timex3_bitmap)
        # print ("raw source bitmap: ", source_bitmap)
        # print ("raw target bitmap: ", target_bitmap)

        # print ("raw first entity bitmap: ", first_entity_bitmap)
        # print ("raw second entity bitmap: ", second_entity_bitmap)
        
    



def inspect(embedding_file, padding_file, entire=False):

    embed = pickle.load(open(embedding_file, 'rb'))
    id_to_word = embed.id_to_word
    word_to_id = embed.word_to_id
    vectors = embed.vectors

    pad = pickle.load(open(padding_file, 'rb'))

    # training data
    # data = pad[0]

    # dev data
    data = pad[1]

    # testing data
    # data = pad[2]

    # closure testing data
    # data = pad[3]
    

    label_count = {}
    
    # label count for training data
    # count = pad[4]
    # for i in range(3):
    #     print ("label " + str(i) + ": " + str(count[i]))

    # dataset count for training data
    count = pad[4]
    for i in range(2):
        print ("label " + str(i) + ": " + str(count[i]))

    
    
    if entire:
        id_list = [i for i in range(len(data[0]))]
    else:
        # id_list = [i for i in range(3)]
        id_list = [0, 1, 2, 3, 4, 5, 10, 15, 22, 100, 137, 159, 1000, 1003]

    for i in id_list:

        list_idx = i
        
        sent_embed = data[0][i]
        # pos_embed_source = data[1][i]
        # pos_embed_target = data[2][i]
        # event_bitmap = data[3][i]
        # timex3_bitmap = data[4][i]
        # source_bitmap = data[5][i]
        # target_bitmap = data[6][i]
        # boolean_features = data[7][i]
        # label = data[8][i]

        event_bitmap = data[1][i]
        timex3_bitmap = data[2][i]
        # source_bitmap = data[3][i]
        # target_bitmap = data[4][i]

        first_entity_bitmap = data[3][i]
        second_entity_bitmap = data[4][i]
        
        label = data[5][i]

        is_thyme = data[6][i]

        label_count[label] = label_count.get(label, 0) + 1
        
        sentence = ''
        event = ''
        timex3 = ''
        # source = ''
        # target = ''

        first_entity = ''
        second_entity = ''
        
        for idx in sent_embed:
            if idx == 0:
                break
            sentence += id_to_word[idx] + ' '

        for i, b in enumerate(event_bitmap):
            if b == 1:
                event += id_to_word[sent_embed[i]] + ' '

        for i, b in enumerate(timex3_bitmap):
            if b == 1:
                timex3 += id_to_word[sent_embed[i]] + ' '


        # for i, b in enumerate(source_bitmap):
        #     if b == 1:
        #         source += id_to_word[sent_embed[i]] + ' '

        for i, b in enumerate(first_entity_bitmap):
            if b == 1:
                first_entity += id_to_word[sent_embed[i]] + ' '

                
        # for i, b in enumerate(target_bitmap):
        #     if b == 1:
        #         target += id_to_word[sent_embed[i]] + ' '

        for i, b in enumerate(second_entity_bitmap):
            if b == 1:
                second_entity += id_to_word[sent_embed[i]] + ' '

        print ("list idx: ", list_idx + 1)
                
        print ("sentence: ", sentence)
        print ("event: ", event)
        print ("timex3: ", timex3)
        # print ("source: ", source)
        # print ("target: ", target)

        print ("first entity: ", first_entity)
        print ("second entity: ", second_entity)
        
        print ("label: ", label)

        print ("is thyme: ", is_thyme)
        # print ("raw event bitmap: ", event_bitmap)
        # print ("raw timex3 bitmap: ", timex3_bitmap)
        # print ("raw source bitmap: ", source_bitmap)
        # print ("raw target bitmap: ", target_bitmap)

        # print ("raw first entity bitmap: ", first_entity_bitmap)
        # print ("raw second entity bitmap: ", second_entity_bitmap)
        
    print ("label count")
    print (label_count)

if __name__ == "__main__":
    # inspect('/home/yuyi/cs6890/project/data/embedding_with_xml_tag.pkl', '/home/yuyi/cs6890/project/data/padding_test_event_vs_time_with_xml_tag.pkl')
    # inspect('/home/yuyi/cs6890/project/data/embedding_with_xml_tag.pkl', '/home/yuyi/cs6890/project/data/padding_event_vs_time_with_xml_tag.pkl', entire=True)

    # inspect('/home/yuyi/cs6890/project/data/embedding_with_xml_tag_4.pkl', '/home/yuyi/cs6890/project/data/padding_event_vs_time_with_xml_tag_4.pkl', entire=True)

    # inspect('/home/yuyi/cs6890/project/data/embedding_test_with_xml_tag_0.pkl', '/home/yuyi/cs6890/project/data/padding_test_event_vs_time_with_xml_tag_0.pkl', entire=True)

    inspect('/home/yuyi/cs6890/project/data/embedding_test_with_xml_tag_1.pkl', '/home/yuyi/cs6890/project/data/step2_padding_test_event_vs_time_with_xml_tag_1.pkl', entire=True)
    
    # inspect('/home/yuyi/cs6890/project/data/embedding_with_xml_tag.pkl', '/home/yuyi/cs6890/project/data/padding_test_event_vs_time_with_xml_tag.pkl', entire=True)        
    # inspect('embedding.pkl', '/home/yuyi/cs6890/project/data/padding_test_event_vs_time.pkl', entire=True)




import xml.etree.ElementTree as ET
import re

from os import listdir
from os.path import isfile, join



class Entity:

    # tid is used for brat annotation format
    # spans: [(start1, end1), (start2, end2), ...]
    # texts: [text1, text2, ...]
    def __init__(self, id_, spans, type_, texts=[], tid=0):
        self.id_ = id_
        self.spans = spans
        self.type_ = type_
        self.texts = texts
        self.tid = tid


class Relation:

    # rid is used for brat annotation format
    def __init__(self, id_, source_id, target_id, type_, rid=0, source_tid=0, target_tid=0):
        self.id_ = id_
        self.source_id = source_id
        self.target_id = target_id
        self.type_ = type_
        self.rid = rid
        self.source_tid = source_tid
        self.target_tid = target_tid


class ParseXMLFile:


    def process(self, dir_to_anno_file, dir_to_raw_file):
        anno_train_dir = dir_to_anno_file + 'train/'
        anno_dev_dir = dir_to_anno_file + 'dev/'
        anno_test_dir = dir_to_anno_file + 'test/'

        raw_train_dir = dir_to_raw_file + 'train/'
        raw_dev_dir = dir_to_raw_file + 'dev/'
        raw_test_dir = dir_to_raw_file + 'test/'
        
        anno_train_file_list = self.collect_files_from_dir(anno_train_dir)
        anno_dev_file_list = self.collect_files_from_dir(anno_dev_dir)
        anno_test_file_list = self.collect_files_from_dir(anno_test_dir)

        self.convert_file_list(anno_train_dir, raw_train_dir, anno_train_file_list)
        self.convert_file_list(anno_dev_dir, raw_dev_dir, anno_dev_file_list)
        self.convert_file_list(anno_test_dir, raw_test_dir, anno_test_file_list)
        


    def convert_file_list(self, dir_to_anno_file, dir_to_raw_file, file_list):
        for file_ in file_list:
            anno_file_path = dir_to_anno_file + file_
            raw_file_path = dir_to_raw_file + file_[:file_.find('.')]
            self.convert(anno_file_path, raw_file_path)
        
        
        

    def convert(self, path_to_anno_file, path_to_raw_file):

        entity_list, relation_list = self.collect_anno(path_to_anno_file)
        entity_list = self.entity_augment(entity_list, path_to_raw_file)
        relation_list = self.relation_augment(entity_list, relation_list)

        output_file_path = path_to_raw_file + '.ann'
        self.output_brat_anno(entity_list, relation_list, output_file_path)


        
        # print ("entity list [0] text: ", entity_list[0].texts)
        # for i in range(50):
        #     print (entity_list[i].texts)
        #     print (entity_list[i].tid)

        # for i in range(10):
        #     print (relation_list[i].rid)
        #     print (relation_list[i].source_tid)
        #     print (relation_list[i].target_tid)




    def collect_files_from_dir(self, path_to_folder):
        files = [f for f in listdir(path_to_folder) if isfile(join(path_to_folder, f))]
        files.sort()
        # print ("files: ", files)
        # print ("files[0]: ", files[0])
        # print ("files[0] type: ", type(files[0]))
        return files

        

    def output_brat_anno(self, entity_list, relation_list, path_to_output_file):
        buffer_ = ''

        for entity in entity_list:
            tmp = ''
            spans = entity.spans
            for span in spans:
                tmp += str(span[0]) + ' ' + str(span[1])
                if span != spans[-1]:
                    tmp += ';'
            tmp += '\t'
            texts = entity.texts
            for text in texts:
                tmp += text
                if text != texts[-1]:
                    text += ' '
                    
            tmp = 'T' + str(entity.tid) + '\t' + entity.type_ + ' ' + tmp + '\n'
            buffer_ += tmp
            
        # for relation in relation_list:

        for relation in relation_list:
            buffer_ += 'R' + str(relation.rid) + '\t' + relation.type_ + ' ' + 'Arg1:T' + str(relation.source_tid) \
                       + ' Arg2:T' + str(relation.target_tid) + '\n'
        
            
        with open(path_to_output_file, 'w') as f:
            f.write(buffer_)

        

            

    def get_entity_tid(self, entity_list, entity_id):
        for entity in entity_list:
            if entity.id_ == entity_id:
                return entity.tid
        

    # add rid, source_tid and target_tid to each relation
    def relation_augment(self, entity_list, relation_list):
        for i in range(len(relation_list)):
            relation = relation_list[i]
            source_tid = self.get_entity_tid(entity_list, relation.source_id)
            target_tid = self.get_entity_tid(entity_list, relation.target_id)
            relation.rid = i + 1
            relation.source_tid = source_tid
            relation.target_tid = target_tid
        return relation_list

        
        
    # add tid and text info to each entity
    def entity_augment(self, entity_list, path_to_raw_file):
        
        with open(path_to_raw_file, 'r') as f:
            for entity in entity_list:
                texts = []
                spans = entity.spans
                # print ("id: ", entity.id_)
                # print ("spans: ", spans)
                for i, span in enumerate(spans):
                    # print ("span: ", span)
                    offset = span[0]
                    size = span[1] - span[0]
                    # print ("offset: ", offset)
                    # print ("size: ", size)
                    f.seek(offset)
                    text = f.read(size)

                    if text.strip() != text:
                        m = re.match('\n*(.*)\n*', text)
                        text = m.group(1)
                        entity.spans[i] = (m.span(1)[0] + offset, m.span(1)[1] + offset)
                    texts.append(text)
                        
                entity.texts = texts
            for i in range(len(entity_list)):
                entity_list[i].tid = i + 1
                
        return entity_list
        

    
    def collect_anno(self, path_to_anno_file):
        
        print ("collecting events...")

        tree = ET.parse(path_to_anno_file)
        root = tree.getroot()
        annotations = root.find('annotations')

        entity_list = []
        relation_list = []
        
        for entity in annotations.findall('entity'):
            span_ = entity.find('span').text

            spans = []
            span_l = span_.split(';')
            for span in span_l:
                pos = span.split(',')
                spans.append((int(pos[0]), int(pos[1])))
                
            type_ = entity.find('type').text
            id_ = entity.find('id').text
            entity_list.append(Entity(id_, spans, type_))


            

        for relation in annotations.findall('relation'):

            # print ("one relation.")
            type_ = relation.find('type').text
            id_ = relation.find('id').text
            properties = relation.find('properties')
            if (type_ == 'TLINK'):
                source_id = properties.find('Source').text
                Type_ = properties.find('Type').text
                target_id = properties.find('Target').text

                relation_list.append(Relation(id_, source_id, target_id, Type_))

        # event_list.sort()
        # timex3_list.sort()


        return entity_list, relation_list


# testing
def main():

    # f = open('/home/yuyi/corpora/THYME/thymedata-master/coloncancer/train/ID001_clinic_001.Temporal-Relation.gold.completed.xml')
    anno_file_path = '/home/yuyi/corpora/THYME/thymedata-master/coloncancer/train/ID001_clinic_001.Temporal-Relation.gold.completed.xml'
    raw_file_path = '/home/yuyi/corpora/THYME/train/ID001_clinic_001'

    anno_file_dir = '/home/yuyi/corpora/THYME/thymedata-master/coloncancer/'
    raw_file_dir = '/home/yuyi/corpora/THYME/'
    
    PF = ParseXMLFile()
    # entity_list, relation_list = PF.collect_anno(file_path)
    # print ("entity list length: ", len(entity_list))
    # print ("relation list length: ", len(relation_list))

    # PF.convert(anno_file_path, raw_file_path)

    PF.process(anno_file_dir, raw_file_dir)

    

if __name__ == '__main__':
    main()

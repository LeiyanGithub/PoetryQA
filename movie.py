# -*- coding: utf-8 -*-
# import fool
import numpy
import re
import os
import jieba
import pandas as pd
import pickle
import json
from pandas.core.frame import DataFrame
from py2neo import Graph
import jieba.posseg as pseg

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

reg = re.compile("[^\u4e00-\u9fa5]")
jieba.load_userdict('dictionary.txt')
count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()

class Movie():
    def __init__(self):
        self.query = ''
        self.vocabulary = {}
        self.person_names = []
        self.movie_names = []
        self.genre_names = []
        self.get_graph()
        self.stopwords = self.get_data('stopwords.txt')
        self.train_data_path = 'train_data.csv'
        self.pos = ['pnt', 'vnt', 'nr', 'dyn', 'ptn', 'gnt']
        self.file_to_id = self.get_json('fileName_to_id.txt')

    
    def get_data(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = f.read()
        return data
    
    def get_json(self, filepath):
        with open(filepath, "r", encoding='utf-8') as f:
            data = json.load(f)
        # print("data: ", data)
        return data

    def get_graph(self):
        self.graph = Graph(
            "http://localhost/",
            port= 7474,
            username="neo4j",
            password="123456"
        )
    
    def export_object(self, object, properties):
        sql = 'Match(p: {}) return '.format(object)
        for ele in properties:
            sql += 'p.{},'.format(ele)
        sql = sql[: -1]
        datas = self.graph.run(sql).data()
        tmp = []
        for data in datas:
            for ele in properties:
                tmp.append(data['p.{}'.format(ele)])
        result = filter(None, [reg.sub('', ele) for ele in filter(None, tmp)])
        return set(result)
    
    def export_dicts(self):
        poetry_names = self.export_object('Poetry', ['poetryName'])
        verse_contents = self.export_object('Verse', ['verseContext'])
        people_names = self.export_object('People', ['peopleName', 'peopleNickname'])
        dynasty_names = self.export_object('Dynasty', ['dynastyName'])
        poetrything_names = self.export_object('Poetrything', ['poetrythingName'])
        genre_names = self.export_object('Genre', ['genreName'])
        with open('dict.txt', 'w+', encoding='utf-8') as f:
            for poetry in poetry_names:
                f.write(poetry + ' pnt\n')
            for verse in verse_contents:
                f.write(verse + ' vnt\n')
            for people in people_names:
                f.write(people + ' nr\n')
            for dynasty in dynasty_names:
                f.write(dynasty + ' dyn\n')
            for poetrything in poetrything_names:
                f.write(poetrything + ' ptn\n')
            for genre in genre_names:
                f.write(genre + ' gnt\n')
        f.close()

    def prepare_train_data(self, filepath):
        files = os.listdir(filepath)
        print("files: ",files)
        index = 0
        train_datas = []
        train_labels = []
        for filename in files:
            index += 1 
            f = open(filepath + filename, 'r', encoding='utf-8')
            lines = f.readlines()
            for line in lines:
                train_datas.append(line.replace('\n', ''))
                train_labels.append(index)
        trains = {'data': train_datas, 'label': train_labels}
        df = DataFrame(trains)
        df['cut_data'] = df['data'].apply(lambda x : " ".join([w for w in list(jieba.cut(x)) if w not in self.stopwords]))
        df.to_csv(self.train_data_path, index=False)
        file_to_id = {}
        for id in range(len(files)):
            file_to_id[id+1] = files[id]
        print("file_to_id: ", file_to_id)
        with open("fileName_to_id.txt", "w", encoding='utf-8') as f:
            f.write(json.dumps(file_to_id, ensure_ascii=False))
        f.close()

    def NB(self):
        df = pd.read_csv(self.train_data_path)
        X_train, X_test, y_train, y_test = train_test_split(df['cut_data'], df['label'], test_size = 0.1, random_state = 0)
        X_train_counts = count_vect.fit_transform(X_train)
        # print("vocabulary: ", count_vect.vocabulary_)
        # print('x_train_counts: ', X_train_counts.toarray())
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        # print('x_train_tfidf: ', X_train_tfidf)
        # self.clf = MultinomialNB().fit(X_train_tfidf, y_train)
        self.clf = LinearSVC().fit(X_train_tfidf, y_train)
        pickle.dump(self.clf, open("models/train_model.pkl", "wb"))
        pickle.dump(count_vect, open("models/CountVectorizer.pkl", 'wb'))
        pickle.dump(tfidf_transformer, open("models/TfidfTransformer.pkl", 'wb'))
    
    def prepare_query(self, query):
        pos_query = pseg.cut(query, use_paddle=True)
        format_query = []
        for word, flag in pos_query:
            if word not in self.stopwords:
                if flag not in self.pos:
                    format_query.append(word)
                else:
                    format_query.append(flag)
                    self.entity = word
        return " ".join(format_query)
    
    def predict(self, query):
        format_query = self.prepare_query(query)
        print('format_query: ', format_query)
        pred_cat_id=self.clf.predict(self.count_vect.transform([format_query]))
        # print(pred_cat_id[0])
        return pred_cat_id[0]
    
    def sql(self, entity, id):
        print("id: ", id)
        print("entity: ", entity)
        # print("file_to_id: ", self.file_to_id)
        key = self.file_to_id[str(id)]
        # print("key: ", key)
        cypher = {
            "人-介绍.txt": 'match (m: People) where m.peopleName = "{}" or m.peopleNickname = "{}" return m.peopleIntroduction'.format(entity, entity),
            "人-出生地.txt": 'match (m: People) where m.peopleName = "{}" or m.peopleNickname = "{}" return m.peopleBurnPlace'.format(entity, entity),
            "人-成就.txt": 'match (m: People) where m.peopleName = "{}" or m.peopleNickname = "{}" return m.peopleAchievements'.format(entity, entity),
            "人-昵称.txt": 'match (m: People) where m.peopleName = "{}" or m.peopleNickname = "{}" return m.peopleNickname'.format(entity, entity),
            "人-经历.txt": 'match (m: People) where m.peopleName = "{}" or m.peopleNickname = "{}" return m.peopleAnecodate'.format(entity, entity),
            "诗-作者.txt": 'match (m: Poetry) where m.poetryName = "{}" return m.poetryAuthor'.format(entity),
            "诗-内容.txt": 'match (m: Poetry) where m.poetryName = "{}" return m.poetryContext'.format(entity),
            "诗-属于-状物.txt": 'MATCH (p)-[r:poetryBelongGenreTo]->(q) where p.poetryName = "{}" RETURN q.genreName'.format(entity),
            "诗-提及-人物.txt": 'MATCH (p)-[r:poetryMentionPeopleTo]->(q) where p.poetryName = "{}" RETURN q.peopleName'.format(entity),
            "诗-提及-地点.txt": 'MATCH (p)-[r:poetryMentionPlaceTo]->(q) where p.poetryName = "{}" RETURN q.locationName'.format(entity),
            "诗-提及-朝代.txt": 'MATCH (p)-[r:poetryMentionDynastyTo]->(q) where p.poetryName = "{}" RETURN q.dynastyName'.format(entity),
            "诗-提及-诗词.txt": 'MATCH (p)-[r:poetryMentionThingTo]->(q) where p.poetryName = "{}" RETURN q.poetrythingName, q.poetrythingMeaning'.format(entity),            
            "诗-朝代.txt": 'match (m: Poetry) where m.poetryName = "{}" return m.dynasty'.format(entity),
            "诗-类别.txt": 'match (m: Poetry) where m.poetryName = "{}" return m.poetryClass'.format(entity),
            "诗-翻译.txt": 'match (m: Poetry) where m.poetryName = "{}" return m.poetryTranslation'.format(entity),
            "诗-背景.txt": 'match (m: Poetry) where m.poetryName = "{}" return m.poetryBackground'.format(entity),
            "诗-鉴赏.txt": 'match (m: Poetry) where m.poetryName = "{}" return m.poetryAppreciation'.format(entity),
            "诗句-上一句-诗句.txt": 'MATCH (p)-[r:verseBeforeTo]->(q) where q.verseContext = "{}" RETURN p.verseContext'.format(entity),
            "诗句-下一句-诗句.txt": 'MATCH (p)-[r:verseNextTo]->(q) where q.verseContext = "{}" RETURN p.verseContext'.format(entity),
            "诗句-作者.txt": 'match (m: Verse) where m.verseContext = "{}" return m.verseAuthor'.format(entity),
            "诗句-原诗.txt": 'match (p:Poetry), (q:Verse) where p.poetryId = q.poetryId and q.verseContext = "{}" return p.poetryName, p.poetryContext'.format(entity),
            "诗句-提及-人物.txt": 'MATCH (p)-[r:verseMentionPeopleTo]->(q) where p.verseContext = "{}" RETURN q.peopleName'.format(entity),
            "诗句-提及-地点.txt": 'MATCH (p)-[r:verseMentionPlaceTo]->(q) where p.verseContext = "{}" RETURN q.locationName'.format(entity),
            "诗句-提及-诗中的词.txt": 'MATCH (p)-[r:verseMentionThingTo]->(q) where p.verseContext = "{}" RETURN q.poetrythingName,q.poetrythingMeaning'.format(entity),
            "诗句-朝代.txt": 'match (m: Verse) where m.verseContext = "{}" return m.verseDynasty'.format(entity),
            "诗句-翻译.txt": "don't know",
            "诗词-词义.txt": 'match (p:Poetrything) where p.poetrythingName = "{}" return p.poetrythingMeaning'.format(entity)
        }
        return cypher[key]

    def get_answer(self, query_list):
        self.clf = pickle.load(open("models/train_model.pkl", "rb"))
        self.count_vect = pickle.load(open("models/CountVectorizer.pkl", "rb"))
        result = []
        for query in query_list:
            id = self.predict(query)
            entity = self.entity
            sql = self.sql(entity, id)
            # print("entity: ", entity, "id: ", id, "sql: ", sql)
            answer = self.graph.run(sql).data()
            print("query: ", query)
            print("sql: ", sql)
            print("answer: ", answer)
            result.append(answer)
        return result

if __name__ == '__main__':
    movie = Movie()
    # movie.export_dicts()
    # movie.prepare_train_data('template/')
    # movie.NB()
    query_list = [
        "《九月十日即事》的作者是谁",
        "《九月十日即事》的全诗内容有",
        "《九月十日即事》描写的哪种状物",
        "《九月十日即事》的提到了哪些人",
        "《九月十日即事》的提到了哪些地点",
        "《九月十日即事》涉及到朝代有",
        "《九月十日即事》提及到的诗词有哪些？",
        "《九月十日即事》写于什么朝代",
        "《九月十日即事》是什么类型的诗",
        "《九月十日即事》的翻译是",
        "《九月十日即事》的写作背景是",
        "如何鉴赏《九月十日即事》",
        "江州司马青衫湿 的上一句是？",
        "江州司马青衫湿 的下一句是？",
        "江州司马青衫湿 的作者是？",
        "江州司马青衫湿 的原诗是？",
        "江州司马青衫湿 提到的人有",
        "江州司马青衫湿 提到哪些地方？",
        "江州司马青衫湿 涉及到的诗词有",
        "江州司马青衫湿 写于哪个朝代？",
        "介绍一下诗人杜甫",
        "诗人李维出生在哪？",
        "杜甫有哪些成就",
        "杜甫有哪些称呼",
        "杜甫的个人经历有"   
    ]
    movie.get_answer(query_list)
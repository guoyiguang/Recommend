#-*- coding: utf-8 -*-
'''
Created on 2015-06-22

@author: Lockvictor
'''
import sys
import random
import math
import os
from operator import itemgetter
from collections import defaultdict

random.seed(0)

'''
users.dat 数据集 
用户id 用户性别 用户年龄 用户职业 用户所在地邮编
1::F::1::10::48067
2::M::56::16::70072
3::M::25::15::55117

movies.dat 数据集
电影id 电影名称 电影类型 
250::Heavyweights (1994)::Children's|Comedy
251::Hunted, The (1995)::Action
252::I.Q. (1994)::Comedy|Romance

ratings.dat 数据集
用户id 电影id 用户评分  时间戳
157::3519::4::1034355415
157::2571::5::977247494
157::300::3::977248224

'''

'''基于物品协同过滤算法'''
class ItemBasedCF(object):
    ''' TopN recommendation - Item Based Collaborative Filtering '''

    def __init__(self):
        self.trainset = {} # 存储训练集
        self.testset = {} # 存储测试集

        self.n_sim_movie = 20 # 定义相似电影数20
        self.n_rec_movie = 10 # 定义推荐电影数10

        self.movie_sim_mat = {} # 存储电影相似矩阵
        self.movie_popular = {}
        self.movie_count = 0

        print('Similar movie number = %d' % self.n_sim_movie, file=sys.stderr)
        print('Recommended movie number = %d' %
              self.n_rec_movie, file=sys.stderr)

    @staticmethod
    def loadfile(filename):
        ''' 只读的方式打开文件 '''
        fp = open(filename, 'r')
        # enumerate()为枚举，i为行号从0开始，line为值
        for i, line in enumerate(fp):
            # yield 迭代去下一个值，类似next()
            # line.strip()用于去除字符串头尾指定的字符。
            yield line.strip('\r\n')
            if i % 100000 == 0:
                print ('loading %s(%s)' % (filename, i), file=sys.stderr)
        fp.close()
        print ('load %s succ' % filename, file=sys.stderr)

    def generate_dataset(self, filename, pivot=0.7):
        ''' 加载数据并且将数据划分为训练集和测试集'''
        trainset_len = 0
        testset_len = 0

        for line in self.loadfile(filename):
            user, movie, rating, _ = line.split('::')
            # 按照pivot=0.7 比例划分
            if random.random() < pivot:
                '''训练集 数据为 {userid,{moveiesid,rating。。。。}}'''
                self.trainset.setdefault(user, {})
                self.trainset[user][movie] = int(rating)
                trainset_len += 1
            else:
                '''测试集 数据为 {userid,{moveiesid,rating....}}'''
                self.testset.setdefault(user, {})
                self.testset[user][movie] = int(rating)
                testset_len += 1

        print ('划分数据集成功', file=sys.stderr)
        print ('训练集 = %s' % trainset_len, file=sys.stderr)
        print ('测试集 = %s' % testset_len, file=sys.stderr)

    def calc_movie_sim(self):
        ''' 记录电影被观看过的次数，从而反映出电影的流行度'''
        print('counting movies number and popularity...', file=sys.stderr)
        # 计算电影的流行度 实质是一个电影被多少用户操作过（即看过，并有评分） 以下数据虚构
        '''{'914': 23, '3408': 12, '2355': 4, '1197': 12, '2804': 31, '594': 12  .....}'''
        for user, movies in self.trainset.items():
            for movie in movies:
                # count item popularity
                if movie not in self.movie_popular:
                    # 如果电影第一次出现 则置为0 假如到字典中。
                    self.movie_popular[movie] = 0
                self.movie_popular[movie] += 1

        print('count movies number and popularity succ', file=sys.stderr)

        # 计算总共被看过的电影数
        self.movie_count = len(self.movie_popular)
        print('total movie number = %d' % self.movie_count, file=sys.stderr)

        # 根据用户使用习惯 构建物品相似度
        itemsim_mat = self.movie_sim_mat
        print('building co-rated users matrix...', file=sys.stderr)
        '''以下为数据格式，通过for循环依次遍历训练集，找到两个电影之间，如果被一个人同时看过，则累加一'''
        #{'914': defaultdict( <class 'int'>, {'3408': 1, '2355': 1  , '1197': 1, '2804': 1, '594': 1, '919': 1})}
        for user, movies in self.trainset.items():
            for m1 in movies:
                itemsim_mat.setdefault(m1, defaultdict(int))
                for m2 in movies:
                    if m1 == m2:
                        continue
                    itemsim_mat[m1][m2] += 1

        print('build co-rated users matrix succ', file=sys.stderr)

        # calculate similarity matrix
        print('calculating movie similarity matrix...', file=sys.stderr)
        simfactor_count = 0
        PRINT_STEP = 2000000

        # 计算用户相似度矩阵
            # 先取得特定用户
        for m1, related_movies in itemsim_mat.items():
            for m2, count in related_movies.items():
                # 以下公式为 两个a,b电影共同被喜欢的用户数/ 根号下（喜欢电影a的用户数 乘 喜欢电影 b的用户数）
                itemsim_mat[m1][m2] = count / math.sqrt(
                    self.movie_popular[m1] * self.movie_popular[m2])
                simfactor_count += 1
                if simfactor_count % PRINT_STEP == 0:
                    print('calculating movie similarity factor(%d)' %
                          simfactor_count, file=sys.stderr)

        print('calculate movie similarity matrix(similarity factor) succ',
              file=sys.stderr)
        print('Total similarity factor number = %d' %
              simfactor_count, file=sys.stderr)
    '''找到K个相似电影 并推荐N个电影'''
    def recommend(self, user):

        K = self.n_sim_movie
        N = self.n_rec_movie
        rank = {}
        watched_movies = self.trainset[user]

        for movie, rating in watched_movies.items():
            # 对于用户看过的每个电影 都找出其相似度最高的前K个电影
            for related_movie, similarity_factor in sorted(self.movie_sim_mat[movie].items(),
                                                           key=itemgetter(1), reverse=True)[:K]:
                if related_movie in watched_movies:
                    continue
                rank.setdefault(related_movie, 0)
                # 假如评分权重，
                rank[related_movie] += similarity_factor * rating
        # 返回综合评分最高的N个电影
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]
    # 计算准确率，召回率，覆盖率，流行度
    def evaluate(self):
        ''' print evaluation result: precision, recall, coverage and popularity '''
        print('Evaluation start...', file=sys.stderr)

        N = self.n_rec_movie
        #  varables for precision and recall
        hit = 0
        rec_count = 0
        test_count = 0
        # varables for coverage
        all_rec_movies = set()
        # varables for popularity
        popular_sum = 0

        for i, user in enumerate(self.trainset):
            if i % 500 == 0:
                print ('recommended for %d users' % i, file=sys.stderr)
            test_movies = self.testset.get(user, {})
            rec_movies = self.recommend(user)
            for movie, _ in rec_movies:
                if movie in test_movies:
                    hit += 1
                all_rec_movies.add(movie)
                popular_sum += math.log(1 + self.movie_popular[movie])
            rec_count += N
            test_count += len(test_movies)
        # 准确率 = 推荐中的电影/总推荐的电影
        precision = hit / (1.0 * rec_count)
        # 召回率 = 推荐中的电影/测试集中所有电影数目
        recall = hit / (1.0 * test_count)
        coverage = len(all_rec_movies) / (1.0 * self.movie_count)
        popularity = popular_sum / (1.0 * rec_count)

        print ('precision=%.4f\trecall=%.4f\tcoverage=%.4f\tpopularity=%.4f' %
               (precision, recall, coverage, popularity), file=sys.stderr)


if __name__ == '__main__':
    ratingfile = os.path.join('ml-1m', 'ratings.dat')
    itemcf = ItemBasedCF()
    itemcf.generate_dataset(ratingfile)
    itemcf.calc_movie_sim()
    print(itemcf.recommend('1'))
    #itemcf.evaluate()

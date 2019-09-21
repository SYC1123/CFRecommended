# coding:UTF-8
import numpy as np
from user_based_recommend import load_data, similarity, top_k


def item_based_recommend(data, w, user):
    '''
    基于商品相似度为用户user推荐商品
    :param data: （mat）商品用户矩阵
    :param w: （mat）商品与商品之间的相似性
    :param user: （int）用户的编号
    :return: （list）推荐列表
    '''
    m, n = np.shape(data)  # m=商品数量 n=用户数量
    interaction = data[:, user].T  # 用户user的互动商品信息

    # 1.找到用户user没有互动的商品
    not_inter = []
    for i in range(n):
        if interaction[0, i] == 0:  # 用户user未打分项
            not_inter.append(i)
    # 2.对没有互动过的商品进行预测
    predict = {}
    for x in not_inter:
        item = np.copy(interaction)  # 获取用户user对商品的互动信息
        for j in range(m):  # 对每一个商品
            if item[0, j] != 0:  # 利用互动过的商品进行预测
                if x not in predict:
                    predict[x] = w[x, j] * item[0, j]
                else:
                    predict[x] = predict[x] + w[x, j] * item[0, j]
    # 按照预测的大小从大到小排序
    return sorted(predict.items(), key=lambda d: d[1], reverse=True)


if __name__ == '__main__':
    # 1.导入用户商品数据
    print('--------1.load data--------')
    data = load_data('data.txt')
    # 将用户商品矩阵转换成商品用户矩阵
    data = data.T
    # 2.计算用户之间的相似性
    print('--------2. calculate similarity between users--------')
    w = similarity(data)
    # 3.利用用户之间的相似性进行推荐
    print('--------3. predict--------')
    predict = item_based_recommend(data, w, 0)
    # 4.进行top_k推荐
    top_recom = top_k(predict, 2)
    print(top_recom)

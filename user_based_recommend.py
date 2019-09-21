import numpy as np


def cos_sim(x, y):
    '''
    余弦相似度
    :param x: （mat）以行向量的形式存储，可以是用户或者商品
    :param y: （mat）以行向量的形式存储，可以是用户或者商品
    :return: x和y之间的余弦相似度
    '''
    numerator = x * y.T  # x和y之间的额内积
    denominator = np.sqrt(x * x.T) * np.sqrt(y * y.T)  # x的范数乘y的范数
    return (numerator / denominator)[0, 0]


def similarity(data):
    '''
    计算矩阵中任意两行之间的相似度
    :param data: （mat）任意矩阵
    :return: w（mat）任意两行之间相似度
    '''
    m = np.shape(data)[0]  # 用户数量
    # 初始化相似度矩阵
    w = np.mat(np.zeros((m, m)))

    for i in range(m):
        for j in range(i, m):
            if j != i:
                # 计算任意两行之间的相似度
                w[i, j] = cos_sim(data[i,], data[j,])
                w[j, i] = w[i, j]
            else:
                w[i, j] = 0  # 相似度矩阵中约定自身相似度的值为0
    return w


def user_based_recommend(data, w, user):
    '''
    基于用户相似性为用户user推荐商品
    :param data: （mat）用户商品矩阵
    :param w: （mat）用户之间的相似度
    :param user: （int）用户的编号
    :return: predict（list）推荐列表
    '''
    m, n = np.shape(data)  # m有几行 n有几列
    interaction = data[user,]  # 用户与商品信息

    # 1.找到用户user没有互动过的商品
    not_inter = []
    for i in range(n):
        if interaction[0, i] == 0:  # 没有互动的商品
            not_inter.append(i)

    # 2.对没有互动过的商品进行预测
    predict = {}
    for x in not_inter:
        item = np.copy(data[:, x])
        for i in range(m):  # 对每一个用户
            if item[i, 0] != 0:  # 若该用户对商品x有过互动
                if x not in predict:
                    predict[x] = w[user, i] * item[i, 0]
                else:
                    predict[x] = predict[x] + w[user, i] * item[i, 0]
    # 3.按照预测的大小从大到小排序
    return sorted(predict.items(), key=lambda d: d[1], reverse=True)


# 转化为商品-用户矩阵



def load_data(file_path):
    '''
    导入用户商品数据
    :param file_path:（string）用户商品数据存放的文件
    :return: 用户商品矩阵
    '''
    f = open(file_path)
    data = []
    for line in f.readlines():
        lines = line.strip().split(',')
        tmp = []
        for x in lines:
            if x != "-":
                tmp.append(float(x))
            else:
                tmp.append(0)
        data.append(tmp)
    f.close()

    return np.mat(data)


def top_k(predict, k):
    '''
    为用户推荐前K个商品
    :param predict: （list）排序好的商品列表
    :param k: （int）推荐的商品个数
    :return: （list）top_k个商品
    '''
    top_recom = []
    len_result = len(predict)
    if k >= len_result:
        top_recom = predict
    else:
        for i in range(k):
            top_recom.append(predict[i])
    return top_recom


if __name__ == '__main__':
    # 1.导入用户商品数据
    print('--------1.load data--------')
    data = load_data('data.txt')
    # 2.计算用户之间的相似性
    print('--------2. calculate similarity between users--------')
    w = similarity(data)
    # 3.利用用户之间的相似性进行推荐
    print('--------3. predict--------')
    predict = user_based_recommend(data, w, 0)
    # 4.进行Top——K推荐
    top_recom = top_k(predict, 2)
    print(top_recom)

import numpy as np
from collections import Counter
import math
import treeplot

class decision_tree():
    def __init__(self, trainset,attr_list,class_name):
        self.trainset=trainset
        self.attr_list=attr_list
        self.train_num=np.size(trainset,0)
        self.attr_num=np.size(trainset,1)-1
        self.class_name=class_name
        self.attr_dict=self.get_attr_dict(trainset,attr_list,self.attr_num)
        self.class_list=self.get_class_list(trainset)
        self.tree={}

        '''
        trainset:训练集（不包括属性行）
        attr_list:属性行
        train_num:样例数量
        attr_num：属性数量
        class_num：分类名
        attr_dict:属性名：【可取范围】
        class_list:分类可取范围
        tree：决策树（利用嵌套字典表示）
        '''

#得到属性字典
    def get_attr_dict(self,trainset,attr_list,attr_num):
        attr_dict={}
        for i in range(attr_num):
            attr_dict[attr_list[i]]=list(set(trainset[:,i]))
        return attr_dict
#得到分类可取范围
    def get_class_list(self,trainset):
        return list(set(trainset[:,-1]))
#判断当前样例的Class是否都一样
    def IsClassAllsame(self,class_array):
        return (len(set(class_array))==1)
# 判断当前样例的属性是否都一样
    def IsAttrAllsame(self,attr_array):
        return len(set([tuple(t) for t in attr_array]))==1
#找到当前样例最多的类别
    def find_most_class(self,class_array):
        return Counter(class_array).most_common(1)[0][0]
#计算信息熵（D为一维数组）
    def Ent(self,D):
        count=Counter(D).values()
        sum_count=sum(count)
        return -sum([i / sum_count * math.log(i / sum_count, 2) for i in count])
#利用ID3算法选出类别
    def get_best_attr(self,example,attr_list):
        pass

#删除二维数组某一列
    def delete_column(self,D,index):
        D_1=D[:,:index]
        D_2=D[:,index+1:]
        return np.concatenate((D_1, D_2), axis=1)
#开始训练
    def TreeGenerate_first(self):
        self.TreeGenerate(self.trainset,self.attr_list,self.tree)
#生成树
    def TreeGenerate(self, example, attr_list, node):
        if self.IsClassAllsame(example[:, -1]):
            node[self.class_name] = example[0, -1]
            return
        if len(attr_list) == 0 or self.IsAttrAllsame(example[:, :-1]):
            node[self.class_name] = self.find_most_class(example[:, -1])
            return

        best_attr_index = self.get_best_attr(example, attr_list)

        best_attr = attr_list[best_attr_index]
        node[best_attr] = {}

        for attr in self.attr_dict[best_attr]:
            node[best_attr][attr] = {}

            Dv = []
            for line in example:
                if line[best_attr_index] == attr:
                    Dv.append(line)
            if len(Dv) == 0:
                node[best_attr][attr][self.class_name] = self.find_most_class(example[:, -1])
            else:
                Dv = np.array(Dv)
                attr_list_new = [attr_list[i] for i in range(len(attr_list)) if i != best_attr_index]
                self.TreeGenerate(self.delete_column(Dv, best_attr_index), attr_list_new, node[best_attr][attr])

#预测
    def predict_1dim(self,test):
        node = self.tree
        while True:
            attr, _ = list(node.items())[0]

            if attr == self.class_name:
                result = node[attr]
                return result
            else:
                attr_index = self.attr_list.index(attr)
                value = test[attr_index]
                node = node[attr][value]


    def predict(self,test):
        assert(test.ndim<=2)
        if test.ndim==1:
            assert(np.size(test)==self.attr_num)
            return self.predict_1dim(test)

        else:
            assert (np.size(test,1) == self.attr_num)
            result=[]
            for line in test:
                result.append(self.predict_1dim(line))
            return  result







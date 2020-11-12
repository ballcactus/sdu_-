from decision_tree import decision_tree
import numpy as np
import treeplot
from collections import Counter

class ID3_tree(decision_tree):
    def __init__(self,trainset,attr_list,class_name):
        super().__init__(trainset,attr_list,class_name)

    def get_best_attr(self, example, attr_list):
        Gain = []
        example_num = np.size(example, 0)
        Ent_D = self.Ent(example[:, -1])
        tmp_D = []
        Ent_Dv = 0

        for i in range(len(attr_list)):
            for attr in self.attr_dict[attr_list[i]]:
                for line in example:
                    if line[i] == attr:
                        tmp_D.append(line[-1])
                Ent_Dv = Ent_Dv + self.Ent(tmp_D) * len(tmp_D) / example_num
                tmp_D.clear()
            Gain.append(Ent_D - Ent_Dv)
            Ent_Dv = 0
        return Gain.index(max(Gain))



class CART(decision_tree):
    def __init__(self,trainset,attr_list,class_name):
        super().__init__(trainset,attr_list,class_name)

    def Gini(self,D):
        count = Counter(D).values()
        sum_count = sum(count)
        P=[(i/sum_count)**2 for i in count]
        return 1-sum(P)

    def get_best_attr(self, example, attr_list,attr_dict=None):

        Gini_index=[]
        tmp_D1=[]
        tmp_D2=[]
        best_index=-1
        best_attr_value=None
        min=1
        example_num=np.size(example,0)
        for i in range(len(attr_list)):
            Gini_index.append({})
            for attr in attr_dict[attr_list[i]]:
                for line in example:
                    if line[i]==attr:
                        tmp_D1.append(line[-1])
                    else:
                        tmp_D2.append(line[-1])
                Gini_index[i][attr]=self.Gini(tmp_D1)*len(tmp_D1)/example_num+self.Gini(tmp_D2)*len(tmp_D2)/example_num
                if min > Gini_index[i][attr]:
                    min=Gini_index[i][attr]
                    best_index=i
                    best_attr_value=attr
                tmp_D1.clear()
                tmp_D2.clear()

        return best_index,best_attr_value,min

    def TreeGenerate_first(self):
        self.TreeGenerate(self.trainset,self.attr_list,self.tree,self.attr_dict,0.1,2)

    def TreeGenerate(self, example, attr_list,node,attr_dict=None,Gini_eps=None,exam_num_eps=None):
        if self.IsClassAllsame(example[:, -1]):
            node[self.class_name] = example[0, -1]
            return
        if len(attr_list) == 0 or self.IsAttrAllsame(example[:, :-1]):
            node[self.class_name] = self.find_most_class(example[:, -1])
            return
        if np.size(example,0)<exam_num_eps or self.Gini(example[:,-1])<Gini_eps:
            node[self.class_name] = self.find_most_class(example[:, -1])
            return


        best_attr_index,best_attr_value,min_gini = self.get_best_attr(example, attr_list,attr_dict)

        best_attr = attr_list[best_attr_index]

        node[(best_attr,best_attr_value)]={}




        D1 = []
        D2=[]
        for line in example:
            if line[best_attr_index] == best_attr_value:
                D1.append(line)
            else:
                D2.append(line)
        if len(D1) == 0:
            node[(best_attr,best_attr_value)]['是'] = self.find_most_class(example[:, -1])
        else:
            D1=np.array(D1)
            tmp_dict=attr_dict.copy()
            tmp_dict.pop(best_attr)
            attr_list_new = [attr_list[i] for i in range(len(attr_list)) if i != best_attr_index]
            node[(best_attr, best_attr_value)]['是']={}
            self.TreeGenerate(self.delete_column(D1, best_attr_index), attr_list_new, node[(best_attr,best_attr_value)]['是'],tmp_dict,Gini_eps,exam_num_eps)

        if len(D2)==0:
            node[(best_attr,best_attr_value)]['否'] = self.find_most_class(example[:, -1])
        else:
            D2=np.array(D2)
            tmp_dict=attr_dict.copy()

            tmp_dict[best_attr]=[i for i in tmp_dict[best_attr] if i!=best_attr_value]
            node[(best_attr, best_attr_value)]['否']={}
            self.TreeGenerate(D2, attr_list,
                              node[(best_attr, best_attr_value)]['否'], tmp_dict, Gini_eps, exam_num_eps)











if __name__ == '__main__':
    trainset=np.array([ ['青绿','蜷缩','浊响','清晰','凹陷','硬滑','好瓜'],
                        ['乌黑','蜷缩','沉闷','清晰','凹陷','硬滑','好瓜'],
                        ['乌黑','蜷缩','浊响','清晰','凹陷','硬滑','好瓜'],
                        ['青绿','蜷缩','沉闷','清晰','凹陷','硬滑','好瓜'],
                        ['浅白','蜷缩','浊响','清晰','凹陷','硬滑','好瓜'],
                        ['青绿','稍蜷','浊响','清晰','稍凹','软粘','好瓜'],
                        ['乌黑','稍蜷','浊响','稍糊','稍凹','软粘','好瓜'],
                        ['乌黑','稍蜷','浊响','清晰','稍凹','硬滑','好瓜'],
                        ['乌黑','稍蜷','沉闷','稍糊','稍凹','硬滑','坏瓜'],
                        ['青绿','硬挺','清脆','清晰','平坦','软粘','坏瓜'],
                        ['浅白','硬挺','清脆','模糊','平坦','硬滑','坏瓜'],
                        ['浅白','蜷缩','浊响','模糊','平坦','软粘','坏瓜'],
                        ['青绿','稍蜷','浊响','稍糊','凹陷','硬滑','坏瓜'],
                        ['浅白','稍蜷','沉闷','稍糊','凹陷','硬滑','坏瓜'],
                        ['乌黑','稍蜷','浊响','清晰','稍凹','软粘','坏瓜'],
                        ['浅白','蜷缩','浊响','模糊','平坦','硬滑','坏瓜'],
                        ['青绿','蜷缩','沉闷','稍糊','稍凹','硬滑','坏瓜'],
                        ])
    attr_list=['色泽','根蒂','敲声','纹理','脐部','触感']
    classname='好坏'

    test=CART(trainset,attr_list,classname)
    test.TreeGenerate_first()
    print(test.tree)
    treeplot.createPlot(test.tree, test.class_name)



'''
本程序采用西瓜数据集3.0a，利用logistics regression算法进行分类
下降方式有Newton与Gradient descent两种方法
PS:初始化会对收敛性造成影响
'''


import numpy as np
import matplotlib.pyplot as plt



class  logistic_reg():

    def __init__(self,train_set,iterThreshold):
#        self.train_set=train_set

        self.train_num=np.size(train_set,0)
        self.attr_num=np.size(train_set,1)-1
        self.beta=0
        self.train_x=np.hstack((train_set[:,:-1],np.ones([self.train_num,1])))
        self.train_y=train_set[:,-1:]
        self.iterThreshold=iterThreshold
        self.beta_t_x=0
        self.loss=0
        '''
        train_num:训练集样例数
        attr_num：属性数
        beta：待训练参数beta  [attr_num+1]
        train_x：训练集属性集合 [train_num,attr_num+1]，最后一列为1
        train_y：训练集target集合 [train_num,1]
        iterThreshold：停止阈值   
        beta_t_x：beta与train_x的乘积 [train_num]
        loss：当前损失
        '''



#计算loss
    def get_loss(self):

        loss=-(self.train_y.reshape(self.train_num)*self.beta_t_x+
        np.log(np.exp(self.beta_t_x)+1)).sum()

        return loss

#计算一阶导数
    def get_first_der(self):
        return -((self.train_y-(np.exp(self.beta_t_x)/(1+np.exp(self.beta_t_x))).reshape(self.train_num,1))*self.train_x).sum(axis=0)

#计算二阶导数（hessian阵）
    def get_second_der(self):

        hessian=np.zeros([self.attr_num+1,self.attr_num+1])
        for i in range(self.train_num):
            xi=self.train_x[i:i+1,:]
            p1_xi=(self.beta*xi.reshape(self.attr_num+1)).sum()
            p1_xi=np.exp(p1_xi)/(1+np.exp(p1_xi))
            hessian=hessian+np.matmul(xi.transpose(1,0),xi)*p1_xi*(1-p1_xi)

        return hessian

#牛顿迭代
    def train_by_newton(self):
        self.beta=np.random.rand(self.attr_num+1)
        i=0

        while True:
            self.beta_t_x = (self.beta * self.train_x).sum(axis=1)
            d_beta = self.get_first_der()
            d2_beta = self.get_second_der()
            self.beta=self.beta-np.matmul(np.linalg.inv(d2_beta),d_beta.reshape(self.attr_num+1,1)).reshape(self.attr_num+1)
            L_last=self.get_loss()

            if np.abs(self.loss-L_last)<self.iterThreshold :
                break
            self.loss = L_last
            print('第'+str(i)+'轮的loss为'+str(self.loss))
            i=i+1

        plt=self.plot_line()
        plt.title("Newton")
        plt.show()

        return self.beta

#梯度下降
    def train_by_GD(self,alpha=5e-2):
        self.beta = np.random.rand(self.attr_num + 1)
        i = 0

        while True:
            self.beta_t_x = (self.beta * self.train_x).sum(axis=1)
            d_beta = self.get_first_der()
            self.beta=self.beta-alpha*d_beta
            L_last = self.get_loss()

            if np.abs(self.loss - L_last) < self.iterThreshold:
                break
            self.loss = L_last
            print('第' + str(i) + '轮的loss为' + str(self.loss))
            i = i + 1

        plt=self.plot_line()
        plt.title("GD")
        plt.show()
        return self.beta

#预测函数，M为[test_num,attr_num]
    def predict(self,M):
        assert (np.size(M,1)==self.attr_num),"属性数错误"

        b_t_x=(self.beta[:2]*M).sum(axis=1)+self.beta[2]

        y=np.zeros(np.size(M,0))
        y[b_t_x>0]=1

        return y #[test_num]




#绘图
    def plot_line(self):
        #绘制训练集
        for i in range(self.train_num):
            if (self.train_y[i]==1):
                plt.scatter(self.train_x[i,0],self.train_x[i,1],c='r')
            else:
                plt.scatter(self.train_x[i, 0], self.train_x[i, 1],c='b')
        #绘制分割线
        x= np.linspace(0,1,100)
        y=(x*self.beta[0]+self.beta[2])/-self.beta[1]
        plt.plot(x, y, '-r')
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False   #显示中文
        plt.xlabel("密度")
        plt.ylabel("含糖量")
        plt.grid()


        return plt

if __name__ == '__main__':
    train_set=np.array([[0.697,0.460,1],
                        [0.774,0.376,1],
                        [0.634,0.264,1],
                        [0.608,0.318,1],
                        [0.556,0.215,1],
                        [0.403,0.237,1],
                        [0.481,0.149,1],
                        [0.437,0.211,1],
                        [0.666,0.091,0],
                        [0.243,0.267,0],
                        [0.245,0.057,0],
                        [0.343,0.099,0],
                        [0.639,0.161,0],
                        [0.657,0.198,0],
                        [0.360,0.370,0],
                        [0.593,0.042,0],
                        [0.719,0.103,0]])
    iterThreshold=1e-6


#梯度下降训练
    GD=logistic_reg(train_set,iterThreshold)
    GD.train_by_GD(alpha=0.05)

    print('随机梯度的参数为'+str(GD.beta))

#牛顿迭代法训练
    Newton=logistic_reg(train_set,iterThreshold)
    Newton.train_by_newton()

    print('牛顿迭代法的参数为'+str(Newton.beta))



    

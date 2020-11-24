import numpy as np
import pandas as pd

#单隐层神经网络
class single_hid_layer_network():
    def __init__(self,trainset,input_num,hidden_num,output_num):
        self.train_x=trainset[:,:-1]
        self.train_y=trainset[:,-1:]
        self.batch_size=np.size(trainset,0)
        self.input_num=input_num
        self.hidden_num=hidden_num
        self.output_num=output_num
        self.W_1=np.zeros([hidden_num,input_num])
        self.W_2=np.zeros([output_num,hidden_num])
        self.b_1=np.zeros([hidden_num,1])
        self.b_2=np.zeros([output_num,1])
        '''
        self.train_x:属性值
        self.train_y:标签值
        self.batch_size:样本总数
        self.input_num：输入神经元数
        self.hidden_num：隐层神经元数
        self.output_num：输出神经元数
        self.W_1:第一层矩阵值
        self.b_1:第一层偏移值
        self.W_2:第二层矩阵值
        self.b_2:第二层偏移值
        
    
        '''
#定义激活函数
    def activate_func(self,x):
        return 1/(1+np.exp(-x))
#得到必要的中间值
    def get_value(self,x):
        z_1=np.dot(self.W_1,x)+self.b_1
        a_1=self.activate_func(z_1)
        z_2=np.dot(self.W_2,z_1)+self.b_2
        a_2 = self.activate_func(z_2)

        return z_1,a_1,z_2,a_2
#参数初始化
    def initialiazation(self):
        self.W_1 = np.random.rand(hidden_num, input_num)
        self.W_2 = np.random.rand(output_num, hidden_num)
        self.b_1=np.random.rand(hidden_num, 1)
        self.b_2=np.random.rand(output_num,1)

# 数据标准化(归一化):        # 标准化
    def normalize(self):
        n=self.input_num
        for i in range(n):
            self.train_x[:, i] = (self.train_x[:, i] - np.mean(self.train_x[:, i])) / (np.std(self.train_x[:, i]) + 1.0e-10)
        return self.train_x

    #得到输出值
    def forward(self,x):
        _,_,_,output=self.get_value(x)
        output=output[0]
        result=[1 if (i>0.5) else 0 for i in output]
        return result


#反向传播
    def back_propagation(self,alpha=0.001,eps=1e-7,cost_func="CrossEntropy"):

        self.initialiazation()
        self.normalize()

        m=self.batch_size
        x=self.train_x.T
        epoch=0
        tmploss=0
        while True:
            epoch=epoch+1

            z_1,a_1,z_2,a_2=self.get_value(x)


            if(cost_func=="CrossEntropy"):

                loss = 1/m*np.sum(-self.train_y.T * np.log(a_2) - (1 - self.train_y.T) * np.log(1-a_2))
                d_z2=(a_2-self.train_y.T)

            elif(cost_func=="MSE"):
                loss=np.sum(np.power(a_2-self.train_y.T,2))
                d_z2 = (a_2 - self.train_y.T)*(a_2)*(1-a_2)
            else:
                exit("无此激活函数")
                return

            d_w2=1/m*np.dot(d_z2,a_1.T)
            d_b2=1/m*np.sum(d_z2,axis=1,keepdims=True)

            d_z1=np.dot(self.W_2.T,d_z2)*(a_1)*(1-a_1)
            d_w1 = 1 / m * np.dot(d_z1,x.T)
            d_b1 = 1 / m * np.sum(d_z1, axis=1, keepdims=True)
            if (abs(tmploss-loss)<eps):
                break

            self.W_1 = self.W_1 - alpha*d_w1
            self.b_1 = self.b_1 - alpha*d_b1
            self.W_2 = self.W_2 - alpha*d_w2
            self.b_2 = self.b_2 - alpha*d_b2

            print("第"+str(epoch)+"轮的loss为"+str(loss))
            tmploss=loss



        result=self.forward(x)
        accuracy=np.sum(1-np.power(result-self.train_y.T,2))/m

        print(accuracy) #准确率


if __name__ == '__main__':
    io='horseColicTraining2.xlsx'

    trainset = pd.read_excel(io, sheet_name=0).as_matrix()

    input_num=np.size(trainset,1)-1
    hidden_num=30
    output_num=1

    horseColic=single_hid_layer_network(trainset,input_num,hidden_num,output_num)
    horseColic.back_propagation(cost_func="CrossEntropy")














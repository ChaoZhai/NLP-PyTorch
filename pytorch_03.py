#!/usr/bin/env python
# coding: utf-8

# ## PyTorch Tutorial 03: Define Networks
# 
# # Overview
# 
# In this tutorial, we explain how to define networks.

# PyTorch中有 Sequential模块定义网络，对于比较复杂的网络 一般使用这种继承nn.module的形式

# In[1]:


import torch
import torch.nn as nn


# x是100乘以10的操作，为了方便假设只有8个观测点(第一个维度作为观测)

# In[3]:


x = torch.rand([8,100,10]).detach()
x


# In[8]:


y = torch.rand(8)
y =(y>0.5).int()
y


# 构建torch中的网络 继承nn.modile的方式
# 
# 由多连接个构成网络_init_中需要 self.first_layer 和self.second_layer
# 
#  self.first_layer 第一层，前一次的输出和后一层的输入 维度上必须要在某种意义上保持一致所以第一个参数是1000，隐藏向量是50个
#  
#  self.second_layer第二层，最终输出是1
#  
#  forward中的参数x，就是每个batch的输入，flatten操作是把后面俩个维度拉直了（100,10）变成了一个1000维的向量
#  
# 
# 

# In[14]:


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.first_layer = nn.Linear(1000,50)
        self.second_layer = nn.Linear(50, 1)
    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=2)
        x = nn.functional.relu(self.first_layer(x))
        x = self.second_layer(x)
        return x


# 实例化网络

# In[15]:


mlp = MLP()
output=mlp(x)


# 1.这个output是随机的
# 
# 2.量纲从理论上讲是负无穷和正无穷之间的（转换成概率的话，可以进行logit或者softmax的操作）
# 
# 

# In[16]:


output


# Embedding就是把比如012345678或者有多少个entity，把它应用成多少维 (4个entity,每一个entity对应100个向量）

# In[17]:


class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(4, 100)
    def forward(self, x):
        return self.embedding(x)


# In[19]:


embedding= Embedding()
embedding_input = torch.tensor([[0,1,0],[2,3,3]])
embedding_output = embedding(embedding_input)


# In[20]:


embedding_output.shape


# In[ ]:


# __init__

10指的是之前Embedding的dim（Embedding的维度），15是隐藏层的维度（h的维度）， num_layers就是这个LSTM一共有多少层

bidirectional=True,对它的方向进行一个指定（前向后输入一遍，然后后向前再输入一遍）

# forward

output指的是它每一个位置的hidden，最终一层它每一个timestep的输出

hidden是最终状态的输出

cell 是里面的一些状态

实际上是有4个hidden，因为num_layers=2，但是每一层有向前和向后的操作


# In[21]:


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(10, 
                           15, 
                           num_layers=2, 
                           bidirectional=True, 
                           dropout=0.1)
    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        return output, hidden, cell


# LSTM中实际上个它观测的是第二位，要做一个permte操作才能跟刚才是一致的
# 
# 把第二维移为第一维，第一维移为第二维，第三维不动

# In[27]:


permute_x= x.permute([1,0,2])
lstm=LSTM()
output_lstm1,output_lstm2,output_lstm3 = lstm(permute_x)


# 第一个是在每一个timestap的hidden的维度，就是sequence length X中的100就留下来了
# 
# 第二个batch size  8也就留下来了
# 
# 第三个是 15乘以2，因为是双向的

# In[31]:


output_lstm1.shape


# 第一个维度是4，最终的输出，返回的不止是最后一层，还有之前一层，因为设置了2层（num_layers=2）
# 15 是hidden dim

# In[32]:


output_lstm2.shape


# In[33]:


output_lstm3.shape


# 在Convolution中，
# 
# __init__
# 
# 调用1d函数，（in_channel,out_channel,kernel的维度）
# 
# 

# In[34]:


class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        self.conv1d = nn.Conv1d(100, 50, 2)
    def forward(self, x):
        return self.conv1d(x)


# In[36]:


conv = Conv()
output= conv(x)


# In[37]:


output.shape


# 

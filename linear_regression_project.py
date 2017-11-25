
# coding: utf-8

# 欢迎来到线性回归项目。
# 
# 若项目中的题目有困难没完成也没关系，我们鼓励您带着问题提交项目，评审会给予您诸多帮助。
# 
# 其中证明题可以提交 pdf 格式，手写后扫描或使用公式编辑器（latex，mathtype）均可行。

# # 1 矩阵运算
# 
# ## 1.1 创建一个 4*4 的单位矩阵

# In[1]:


# 这个项目设计来帮你熟悉 python list 和线性代数
# 你不能调用任何python库，包括NumPy，来完成作业

A = [[1,2,3], 
     [2,3,3], 
     [1,2,5]]

B = [[1,2,3,5], 
     [2,3,3,5], 
     [1,2,5,1]]

#TODO 创建一个 4*4 单位矩阵
I = [[1,0,0,0],
     [0,1,0,0],
     [0,0,1,0],
     [0,0,0,1]]


# ## 1.2 返回矩阵的行数和列数

# In[2]:


# TODO 返回矩阵的行数和列数
def shape(M):
    return len(M),len(M[0])


# In[3]:


# 运行以下代码测试你的 shape 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_shape')


# ## 1.3 每个元素四舍五入到特定小数数位

# In[4]:


# TODO 每个元素四舍五入到特定小数数位
# 直接修改参数矩阵，无返回值
def matxRound(M, decPts=4):
    for i in range(0,len(M)):
        for j in range(0,len(M[0])):
            M[i][j] = round(M[i][j], decPts)


# In[5]:


# 运行以下代码测试你的 matxRound 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_matxRound')


# ## 1.4 计算矩阵的转置

# In[19]:


# TODO 计算矩阵的转置
def transpose(M):
    # *M取出矩阵的每一行，zip(*M)生成返回元组的迭代器
    # 返回的元组是M每一行的第i个元素按行的顺序组成
    # col是元组列表的每一组，list(col)将元组转换成list
    # 这样原矩阵的一列变成新矩阵的相应行，最后组成新的矩阵
    return [list(col) for col in zip(*M)]


# In[20]:


# 运行以下代码测试你的 transpose 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_transpose')


# ## 1.5 计算矩阵乘法 AB

# In[21]:


# TODO 计算矩阵乘法 AB，如果无法相乘则返回None
def matxMultiply(A, B):
    _, na = shape(A)
    mb, _ = shape(B)
    if na!=mb:
        return None
    Bt = transpose(B)
    result = [[sum((a*b) for a,b in zip(row,col)) for col in Bt] for row in A]
    return result


# In[22]:


# 运行以下代码测试你的 matxMultiply 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_matxMultiply')


# ---
# 
# # 2 Gaussign Jordan 消元法
# 
# ## 2.1 构造增广矩阵
# 
# $ A = \begin{bmatrix}
#     a_{11}    & a_{12} & ... & a_{1n}\\
#     a_{21}    & a_{22} & ... & a_{2n}\\
#     a_{31}    & a_{22} & ... & a_{3n}\\
#     ...    & ... & ... & ...\\
#     a_{n1}    & a_{n2} & ... & a_{nn}\\
# \end{bmatrix} , b = \begin{bmatrix}
#     b_{1}  \\
#     b_{2}  \\
#     b_{3}  \\
#     ...    \\
#     b_{n}  \\
# \end{bmatrix}$
# 
# 返回 $ Ab = \begin{bmatrix}
#     a_{11}    & a_{12} & ... & a_{1n} & b_{1}\\
#     a_{21}    & a_{22} & ... & a_{2n} & b_{2}\\
#     a_{31}    & a_{22} & ... & a_{3n} & b_{3}\\
#     ...    & ... & ... & ...& ...\\
#     a_{n1}    & a_{n2} & ... & a_{nn} & b_{n} \end{bmatrix}$

# In[23]:


# TODO 构造增广矩阵，假设A，b行数相同
def augmentMatrix(A, b):
    return [ra + rb for ra,rb in zip(A,b)]


# In[24]:


# 运行以下代码测试你的 augmentMatrix 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_augmentMatrix')


# ## 2.2 初等行变换
# - 交换两行
# - 把某行乘以一个非零常数
# - 把某行加上另一行的若干倍：

# In[25]:


# TODO r1 <---> r2
# 直接修改参数矩阵，无返回值
def swapRows(M, r1, r2):
    M[r1], M[r2] = M[r2], M[r1]


# In[26]:


# 运行以下代码测试你的 swapRows 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_swapRows')


# In[27]:


# TODO r1 <--- r1 * scale， scale!=0
# 直接修改参数矩阵，无返回值
def scaleRow(M, r, scale):
    row, c = shape(M)

    if scale!=0:
        M[r] = [i * scale for i in M[r]]
    else:
        raise ValueError


# In[28]:


# 运行以下代码测试你的 scaleRow 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_scaleRow')


# In[29]:


# TODO r1 <--- r1 + r2*scale
# 直接修改参数矩阵，无返回值
def addScaledRow(M, r1, r2, scale):
    x = [i * scale for i in M[r2]]
    M[r1] = [a + b for a, b in zip(M[r1], x)]


# In[30]:


# 运行以下代码测试你的 addScaledRow 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_addScaledRow')


# ## 2.3  Gaussian Jordan 消元法求解 Ax = b

# ### 提示：
# 
# 步骤1 检查A，b是否行数相同
# 
# 步骤2 构造增广矩阵Ab
# 
# 步骤3 逐列转换Ab为化简行阶梯形矩阵 [中文维基链接](https://zh.wikipedia.org/wiki/%E9%98%B6%E6%A2%AF%E5%BD%A2%E7%9F%A9%E9%98%B5#.E5.8C.96.E7.AE.80.E5.90.8E.E7.9A.84-.7Bzh-hans:.E8.A1.8C.3B_zh-hant:.E5.88.97.3B.7D-.E9.98.B6.E6.A2.AF.E5.BD.A2.E7.9F.A9.E9.98.B5)
#     
#     对于Ab的每一列（最后一列除外）
#         当前列为列c
#         寻找列c中 对角线以及对角线以下所有元素（行 c~N）的绝对值的最大值
#         如果绝对值最大值为0
#             那么A为奇异矩阵，返回None （请在问题2.4中证明该命题）
#         否则
#             使用第一个行变换，将绝对值最大值所在行交换到对角线元素所在行（行c） 
#             使用第二个行变换，将列c的对角线元素缩放为1
#             多次使用第三个行变换，将列c的其他元素消为0
#             
# 步骤4 返回Ab的最后一列
# 
# ### 注：
# 我们并没有按照常规方法先把矩阵转化为行阶梯形矩阵，再转换为化简行阶梯形矩阵，而是一步到位。如果你熟悉常规方法的话，可以思考一下两者的等价性。

# In[31]:


# TODO 实现 Gaussain Jordan 方法求解 Ax = b

""" Gaussian Jordan 方法求解 Ax = b.
    参数
        A: 方阵 
        b: 列向量
        decPts: 四舍五入位数，默认为4
        epsilon: 判读是否为0的阈值，默认 1.0e-16
        
    返回列向量 x 使得 Ax = b 
    返回None，如果 A，b 高度不同
    返回None，如果 A 为奇异矩阵
"""
def gj_Solve(A, b, decPts=4, epsilon = 1.0e-16):
    rA, cA = shape(A)
    rb, cb = shape(b)
    x=[]
    res = []
    # 检查A，b是否行数相同
    if rA != rb:
        # 返回None，如果 A，b 高度不同
        return None
    
    # 构造增广矩阵Ab
    else:
        Ab = augmentMatrix(A, b)        
        # 对于Ab的每一列（最后一列除外）
        for j in range(0,len(Ab[0])-1):
            col_max = 0
            max_i = 0
            # 寻找列c中 对角线以及对角线以下所有元素（行 c~N）的绝对值的最大值
            for i in range(j,rA):
                if abs(Ab[i][j])>col_max:
                    col_max = abs(Ab[i][j])
                    max_i = i
            # 如果绝对值最大值为0
            # 那么A为奇异矩阵，返回None
            if col_max<=epsilon:
                return None
            # 否则
            # 使用第一个行变换，将绝对值最大值所在行交换到对角线元素所在行（行c） 
            else:
                if max_i!=j:
                    # 最大值不是对角线
                    # 使用第一个行变换，将绝对值最大值所在行交换到对角线元素所在行（行c）
                    swapRows(Ab, max_i, j)
                    # 使用第二个行变换，将列c的对角线元素缩放为1
                scaleRow(Ab, j, 1.0/Ab[j][j])
                # 多次使用第三个行变换，将列c的其他元素消为0
                for ind in range(0,len(Ab)):
                    if ind!=j:
                        addScaledRow(Ab, ind, j, -1.0*Ab[ind][j])
        # 返回Ab的最后一列
        for row in Ab:
            x.append(row[-1])
        res.append(x)
        matxRound(res, decPts)
        return transpose(res)


# In[32]:


# 运行以下代码测试你的 gj_Solve 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_gj_Solve')


# ## 2.4 证明下面的命题：
# 
# **如果方阵 A 可以被分为4个部分: ** 
# 
# $ A = \begin{bmatrix}
#     I    & X \\
#     Z    & Y \\
# \end{bmatrix} , \text{其中 I 为单位矩阵，Z 为全0矩阵，Y 的第一列全0}$，
# 
# **那么A为奇异矩阵。**
# 
# 提示：从多种角度都可以完成证明
# - 考虑矩阵 Y 和 矩阵 A 的秩
# - 考虑矩阵 Y 和 矩阵 A 的行列式
# - 考虑矩阵 A 的某一列是其他列的线性组合

# ### 证明：

# 1、根据条件，
# 
# I为单位矩阵，对角线元素全为1
# Z为全0矩阵
# Y的第一列全为零
# 
# 2、假设Y的第一列是A的第i列，
# 
# 则，A的第i列是A的前i-1列的线性组合，则A的列向量线性相关，即存在A的某一列，可以用其他几列（A的前i-1列）线性表出；
# 
# 则A的转置矩阵At的前i-1行乘上相应系数能够将第i行化为0，得到Atn，有一行全0的矩阵行列式值也为0（参考https://zh.wikipedia.org/wiki/%E8%A1%8C%E5%88%97%E5%BC%8F ）
# 所以|Atn|=0；
# 
# 参考上述维基百科中行列式的性质，转置矩阵行列式的值等于原矩阵行列式的值，所以|A|=|At|；
# 并且将一行的k倍加进另一行里，行列式的值不变，Atn是At多次进行“将一行的k倍加进另一行里”这种转换得到的，所以|At|=|Atn|=0；
# 
# 所以就有|A|=|At|=|Atn|=0。
# 
# 3、由非奇异方阵的定义：https://zh.wikipedia.org/wiki/%E9%9D%9E%E5%A5%87%E5%BC%82%E6%96%B9%E9%98%B5
# 
# 若方块矩阵满足行列式的值不为0，则称为非奇异方阵，否则称为奇异方阵。
# 
# 所以A是奇异矩阵。
# 
# 
# 
# 
# 
# PS:
# 
# 根据维基百科讲述的行列式的性质，
# 
# 对于分块的三角矩阵，矩阵的行列式等于对角元素的行列式之乘积。
# 
# 是不是可以直接得到det(A)=det(I)det(Y)
# 
# I为单位矩阵，det(I)=1
# 
# Y的第一列全为零，由行列式的性质det(I)=0
# 
# 所以也能直接得到上述第二步的det(A)=0的结论继而推出A是奇异矩阵。
# 
# 

# ---
# 
# # 3 线性回归: 
# 
# ## 3.1 计算损失函数相对于参数的导数 (两个3.1 选做其一)
# 
# 我们定义损失函数 $E$ ：
# $$
# E = \sum_{i=1}^{n}{(y_i - mx_i - b)^2}
# $$
# 定义向量$Y$, 矩阵$X$ 和向量$h$ :
# $$
# Y =  \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n
# \end{bmatrix}
# ,
# X =  \begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 \\
# \end{bmatrix},
# h =  \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix}
# $$
# 
# 证明：
# $$
# \frac{\partial E}{\partial m} = \sum_{i=1}^{n}{-2x_i(y_i - mx_i - b)}
# $$
# 
# $$
# \frac{\partial E}{\partial b} = \sum_{i=1}^{n}{-2(y_i - mx_i - b)}
# $$
# 
# $$
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix} = 2X^TXh - 2X^TY
# $$

# ### 证明：

# $\\$
# 
# $
# \text{变换后的 }X^T\text{为： }
# $
# 
# $
# X^T =  \begin{bmatrix}
# x_1 & x_2 & ... & x_n\\
# 1 & 1 & ... & 1\\
# \end{bmatrix}
# $
# 
# $\\$
# 
# $ 
# \text{已知： }
# Y =  \begin{bmatrix}
# y_1 \\
# y_2 \\
# ... \\
# y_n
# \end{bmatrix}
# ,
# X =  \begin{bmatrix}
# x_1 & 1 \\
# x_2 & 1\\
# ... & ...\\
# x_n & 1 \\
# \end{bmatrix},
# h =  \begin{bmatrix}
# m \\
# b \\
# \end{bmatrix}
# $
# 
# $
# \text{推导出 }\text{： }
# $
# 
# $
# 2X^TXh - 2X^TY = -2X^T(Y-Xh)
# $
# 
# 
# $\\$
# 
# $
# Y-Xh = \begin{bmatrix}
# y_1 - mx_1 - b \\
# y_2 - mx_2 - b \\
# ... \\
# y_n - mx_n - b \\
# \end{bmatrix}
# $
# 
# 
# 
# 
# $
# \text{所以 }\text{： }
# $
# 
# $
# 2X^TXh - 2X^TY = -2X^T\begin{bmatrix}
# y_1 - mx_1 - b \\
# y_2 - mx_2 - b \\
# ... \\
# y_n - mx_n - b \\
# \end{bmatrix} = \begin{bmatrix}
# \sum_{i=1}^{n}{-2x_i(y_i - mx_i - b)} \\
# \sum_{i=1}^{n}{-2(y_i - mx_i - b)}
# \end{bmatrix}
# $
# 
# $
# \text{然后对定义的损失函数(E)求m、b的偏导 }\text{： }
# $
# 
# 
# $
# \frac{\partial E}{\partial m} = \sum_{i=1}^{n}{2(y_i - mx_i - b)\frac{\partial {(y_i - mx_i - b)}}{\partial m}} $
# 
# $= \sum_{i=1}^{n}{2(y_i - mx_i - b)(-x_i)} $
# 
# $= \sum_{i=1}^{n}{-2x_i(y_i - mx_i - b)}
# $
# 
# 
# 
# $
# \frac{\partial E}{\partial b} = \sum_{i=1}^{n}{2(y_i - mx_i - b) \frac{\partial {(y_i - mx_i - b)}}{\partial b}} $
# 
# $= \sum_{i=1}^{n}{2(y_i - mx_i - b)(-1)} $
# 
# $= \sum_{i=1}^{n}{-2(y_i - mx_i - b)}
# $
# 
# 
# $\\$
# 
# $
# \text{综上所述 }\text{： }
# $
# 
# 
# $
# \frac{\partial E}{\partial m} = \sum_{i=1}^{n}{-2x_i(y_i - mx_i - b)}
# $
# 
# $
# \frac{\partial E}{\partial b} = \sum_{i=1}^{n}{-2(y_i - mx_i - b)}
# $
# 
# 
# 
# $
# \begin{bmatrix}
# \frac{\partial E}{\partial m} \\
# \frac{\partial E}{\partial b} 
# \end{bmatrix} = 2X^TXh - 2X^TY
# $

# ## 3.1 计算损失函数相对于参数的导数（两个3.1 选做其一）
# 
# 我们定义损失函数 $E$ ：
# $$
# E = \sum_{i=1}^{n}{(y_i - mx_i - b)^2}
# $$
# 定义向量$Y$, 矩阵$X$ 和向量$h$ :
# $$
# Y =  \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n
# \end{bmatrix}
# ,
# X =  \begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 \\
# \end{bmatrix},
# h =  \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix}
# $$
# 
# 证明：
# 
# $$
# E = Y^TY -2(Xh)^TY + (Xh)^TXh
# $$ 
# 
# $$
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix}  = \frac{\partial E}{\partial h} = 2X^TXh - 2X^TY
# $$

# TODO 请使用 latex （请参照题目的 latex 写法学习）
# 
# TODO 证明：

# ## 3.2  线性回归
# 
# ### 求解方程 $X^TXh = X^TY $, 计算线性回归的最佳参数 h 
# *如果你想更深入地了解Normal Equation是如何做线性回归的，可以看看MIT的线性代数公开课，相关内容在[投影矩阵与最小二乘](http://open.163.com/movie/2010/11/P/U/M6V0BQC4M_M6V2AOJPU.html)。*

# In[33]:


# TODO 实现线性回归
'''
参数：(x,y) 二元组列表
返回：m，b
'''
def linearRegression(points):
    # 构建 Ax = b 的线性方程
    X = [[points[i][0], 1] for i in range(len(points))]
   
    Y = [[points[i][1]] for i in range(len(points))]

    X_T = transpose(X)
  
    A = matxMultiply(X_T, X)
    
    b = matxMultiply(X_T, Y)
   
    m, b = (i[0] for i in gj_Solve(A, b, decPts=4, epsilon=1.0e-16))
    
    return m, b


# ## 3.3 测试你的线性回归实现

# In[55]:


# TODO 构造线性函数
# y = mx + b
m = 8
b = 7

# TODO 构造 100 个线性函数上的点，加上适当的高斯噪音
import random

x_rand = [random.uniform(-50,50) for i in range(100)]
y_rand = [i * m + b for i in x_rand]

x_gauss = [random.gauss(0, 1) for i in range(100)]
y_gauss = [random.gauss(0, 1) for i in range(100)]

xx = [a + c for a,c in zip(x_rand, x_gauss)]
yy = [a + c for a,c in zip(y_rand, y_gauss)]

points = [(x,y) for x,y in zip(xx, yy)]

#TODO 对这100个点进行线性回归，将线性回归得到的函数和原线性函数比较
m_reg, b_reg = linearRegression(points)

print '原始m,b '
print m, b

print '线性回归后的m,b'
print m_reg, b_reg


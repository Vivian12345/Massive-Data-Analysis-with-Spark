#!/usr/bin/env python
# coding: utf-8

# In[1]:


import findspark
findspark.init()


# In[2]:


from pyspark import SparkConf, SparkContext
conf = SparkConf().setMaster("local").setAppName("PageRank")
sc = SparkContext(conf=conf)


# In[3]:


t_f = sc.textFile("p2p-Gnutella04.txt")
lines = t_f.map(lambda line: line.split("\t"))
pr_rdd = lines.filter(lambda x: x[0][0] != "#")


# # 1.算出每一個page分出去的比例
# EX. 1 -> 2,3,4 即 1分出去給每一個page的比例是1/3

# In[4]:


reducer1 = pr_rdd.groupByKey()
outvotes = reducer1.map(lambda x: (x[0],1/len(x[1])))
dict1 = dict(outvotes.collect())


# In[5]:


N = 10876
init = 1/N


# # 2.做出一個儲存所有page的pagerank的dictionary方便後面查找
# + 一開始初始pagerank都是0.2*1/N

# In[6]:


keys = set(pr_rdd.keys().collect())
values = set(pr_rdd.values().collect())
set1 = keys.union(values)

rj = {i:init*0.2 for i in set1}


# # 3.運算pagerank
# + A = beta*M + (1-beta)*(1/N) = 0.8*M + 0.2*(1/N)
# + pr_matrix輸出結果:("1",("0",0.34234)) -> "0" 給 "1" 0.34234 的 pagerank (只有矩陣M的部分)
# + pr_A輸出結果: ("1", 0.34234) ->方便reduce
# + pr_輸出結果: ("1", 0.63263) -> "1"拿到的所有pagerank (只有矩陣M的部分)
# + pr_final輸出結果: ("1", 0.754564) -> "1"拿到的總pagerank (包刮M和1/N)
# + update rj: 所有page的新pagerank

# In[7]:


pr_matrix = pr_rdd.map(lambda x: (x[1], (x[0],0.8*dict1[x[0]]*init)))
pr_A = pr_matrix.map(lambda x: (x[0], x[1][1]))
pr_ = pr_A.reduceByKey(lambda x,y: x+y)
pr_final = pr_.map(lambda x: (x[0],x[1]+0.2*init))
rj.update(pr_final.collect())


# # 4.adjustment
# + 把rj的value加總
# + 調整的幅度就是 (1-rj的value加總)/總頁數
# + pr_final輸出調整過後的pagerank
# + 把新的pagerank再update到rj裡面

# In[8]:


sum1 = sum(rj.values())
adj = (1-sum1)/N
pr_final = pr_final.map(lambda x: (x[0], x[1]+adj))
rj.update(pr_final.collect())


# # 5.iteration
# + 按照之前的邏輯再跑19個迴圈

# In[9]:


rj1 = {}
rj1.update(rj)
for i in range(19):
    pr_matrix_n = pr_rdd.map(lambda x: (x[1], (x[0],0.8*dict1[x[0]]*rj1[x[0]])))
    pr_A_n = pr_matrix_n.map(lambda x: (x[0], x[1][1]))
    pr_n = pr_A_n.reduceByKey(lambda x,y: x+y).cache()
    pr_final_n = pr_n.map(lambda x: (x[0],x[1]+0.2*init))
    rj1.update(pr_final_n.collect())
    sum2 = sum(rj1.values())
    adj1 = (1-sum2)/N
    pr_final_n = pr_final_n.map(lambda x: (x[0], x[1]+adj1))
    rj1.update(pr_final_n.collect())


# # 6.排序+取小數點(到小數點第六位)

# In[29]:


pr_final_decimal = pr_final_n.map(lambda x: (x[0],("%.7f" % x[1])))


# In[34]:


pr_final_top = pr_final_decimal.sortBy(lambda x: eval(x[1]),ascending=False)


# # 7.找top10
# 1. ('1056', '0.0006321') 
# 2. ('1054', '0.0006292')
# 3. ('1536', '0.0005243')
# 4. ('171', '0.0005120')
# 5. ('453', '0.0004959')
# 6. ('407', '0.0004849')
# 7. ('263', '0.0004797')
# 8. ('4664', '0.0004709')
# 9. ('261', '0.0004630')
# 10. ('410', '0.0004613')

# In[37]:


pr_final_top.take(10)


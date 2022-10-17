#!/usr/bin/env python
# coding: utf-8

from pyspark import SparkConf, SparkContext
conf = SparkConf().setMaster("local").setAppName("Similarity")
sc = SparkContext(conf=conf)


# # 讀取文件

# In[6]:


t_all = sc.wholeTextFiles("C:\\Users\\admin\\athletics\\*")


# In[7]:


t_all.take(1)


# # 一、將Docments轉為Shingles

# ## 1.去除標點符號、空格

# In[157]:


topics = t_all.map(lambda x: (x[0][-7:-4],x[1].translate(str.maketrans("\n"," ",string.punctuation)).strip().replace("  "," ")))


# In[9]:


import string


# In[158]:


topics.take(1)


# ## 2.把每個文件中字分開以便後續製作3-shingles

# In[159]:


topic_split = topics.map(lambda x: (x[0],x[1].split(" ")))


# In[160]:


topic_split.take(1)


# ## 3. 建立一個shingle set

# In[25]:


sh_set = []


# In[161]:


# 每三個字為單位，每次向後一個字，append到shingle set
for c in topic_split.collect()[1]:
    for i in range(0,len(c)-2):
        sh_set.append(c[i:i+3])


# ## 4.讓每個文件跟shingle set作比對

# In[162]:


#製作出每個文件自己部分的shingle
def self_set(x):
    splited = x[1].split(" ")
    self_set = []
    for i in range(0,len(splited)-2):
        self_set.append(splited[i:i+3])
    return (x[0],self_set)


# In[164]:


topics_self = topics.map(self_set)


# In[165]:


topics_self.take(2)


# In[163]:


#和所有文件的shingle作比對(拿每個文件自己的shingle set和所有文件製成的shingle set比對)
def shingles(x):
    shingle = []
    for i in sh_set:
        if i in x[1]:
            shingle.append("1")
        else:
            shingle.append("0")
    return (x[0],shingle)


# In[166]:


topics_shingles = topics_self.map(shingles)


# In[167]:


topics_shingles.take(2)


# # 二、MinHash

# ## 1.做出100個不同的hash functions

# In[47]:


import numpy as np


# In[373]:


#h = ((a*x)+b) % c is the standard, accepted way to generate hash functions.
#Also, a and b should be random numbers from the range 1 to c-1


# In[374]:


#為了確保100的都是不同的function->所有(a,b)的集合裡的個數為100個


# In[183]:


hash_list = [(np.random.randint(low=1, high=3406),np.random.randint(low=1, high=3406)) for i in range(200)]


# In[184]:


hash_set = set(hash_list)


# In[185]:


#製作了200個
len(hash_set)


# In[186]:


valid_hash = list(hash_set)


# In[187]:


#留100個
valid_hash = valid_hash[:100]


# In[176]:


valid_hash


# In[375]:


#定義100個hash function (c = 3407)


# In[188]:


for i in range(100):
    def h(i,x):
        return ((valid_hash[i][0]*x)+valid_hash[i][1])%3407


# In[ ]:


#算出每個文件在100個hash function下hash100次的100個hash value


# ## 2. minhash製作出每個文件的signature

# In[189]:


def minhashing(x):
    signatures = []
    #一次一個hash function
    for i in range(100):
        min_hash = 200
        for c in range(len(x[1])):  #len(x)>30000
            #只有在shingle是"1"的時候，才有hash value
            if x[1][c] == "1":
                # 每次都update最小的min hash value 
                if min_hash > h(i,c):
                    min_hash = h(i,c)
        signatures.append(min_hash)
    return (x[0],signatures)


# In[190]:


minhash = topics_shingles.map(minhashing)


# In[249]:


minhash.take(1)


# # 三、LSH(b=50,r=2)

# ## 1.將100個siganature2個2個一組分成50組

# In[192]:


#將100個siganature2個2個分成50組
def cut_bands(x):
    bands = []
    for i in range(50):
        bands.append((x[1][i*2],x[1][i*2+1]))
    return (x[0],bands)


# In[193]:


bands = minhash.map(cut_bands)


# In[194]:


bands.take(1)


# In[83]:


#hash to the same bucket for ≧ 1 band


# ## 2.製作出每個文件的50組分別被hash到哪一個bucket

# In[195]:


# 製作一個hash function(這裡我讓每一組(a,b)，(a*23+b)%51)
def hash_fun(x):
    bucket_ord = []
    for i in x[1]:
        bucket_ord.append((i[0]*23+i[1])%51)
    return (x[0],bucket_ord)


# In[196]:


#算出每個(a,b)分別被hash到哪個bucket
bands_hash = bands.map(hash_fun)


# In[273]:


bands_hash.take(1)


# # 三、計算相似度
# 
# 邏輯如下:
# + 1.紀錄所有文件的(a,b)->屬於哪一個文件、哪一個band(a)、被hash到哪個bucket(b)->(band的index(a),文件編號，所屬bucket的index(b))
# + 2.每個band自己做比較: 以band index為key，比較不同文件所屬的bucket->如果bucket的index代表這兩個文件在這個band被認為是相似的
# + 3.把每個band認為相似的文件pair加起來(ex.如果對於所有band來說(d1,d2)都是相似的->加起來會是50)就會是兩個文件的相似程度
# + 4.算similarity->把加出來的值除以50

# ## 1.紀錄所有文件的(a,b)

# In[210]:


#屬於哪一個文件、哪一個band、被hash到哪個bucket->(band的index,文件編號，所屬bucket的index)
def index_doc_value(x):
    tripple_list = []
    for i in range(len(x[1])):
        tripple_list.append((i,(x[0],x[1][i])))
    return tripple_list


# In[211]:


bands_filter = bands_hash.flatMap(index_doc_value)


# In[212]:


bands_filter.take(2)


# ## 2.每個band自己做比較

# In[215]:


#1.以band index為key分群
candidate_filter = bands_filter.groupByKey().map(lambda x: (x[0],list(x[1])))


# In[277]:


candidate_filter.take(1)


# In[349]:


# 2.比較不同文件所屬的bucket->如果bucket的index代表這兩個文件在這個band被認為是相似的
def pair_collection(x):
    pair_collection = []
    #同一個band
    #不同文件屬於這個band的部分作比較
    for i in range(len(x[1])):
        for c in range(len(x[1])):
            if i >= c:
                continue
            #同一個bucket -> 在這個band被認為是相似的
            elif x[1][i][1] == x[1][c][1]:
                pair_collection.append((x[1][i][0],x[1][c][0]))
    return pair_collection


# ## 3.把每個band認為相似的文件pair加起來就會是兩個文件的相似程度
# + ex.如果對於所有band來說(d1,d2)都是相似的->加起來會是50

# ### (1)method 1

# In[369]:


# 用Counter在計算在每個band裡面的相似pair的個數(不管是哪一個candidate pair在同一個band裡的Counter都只會是1)
row_count = candidate_filter.map(pair_collection).map(lambda x: Counter(x))


# In[370]:


# 把每一個band的Counter加起來
all_count = a.reduce(lambda x,y: x+y)


# In[371]:


all_count


# ### (2)method 2

# In[372]:


# 用Flatmap把每一個band的candidate pair放在同一地方(我們不在乎candidate pair是哪一個band，只在乎個數)
candidate_filter1 = candidate_filter.map(pair_collection).flatMap(lambda x: x)


# In[307]:


candidate_filter1.take(1)


# In[303]:


resume = list(candidate_filter1.collect())


# In[312]:


# 算出candidate pair分別被多少個band認為是相似的
Count = Counter(resume)


# In[313]:


Count


# ## 4.算出Similarity

# In[331]:


# 去除掉被少於2個band認為是相似的pair(也就是只有一個band認為他們是相似的)->candidate pair
Ultimate_candidate = dict([(k,v) for k, v in filter(lambda x: x[1]>=2, list(Count.items()))])


# In[333]:


# 將每個candidate pair被認為是相似的數量 除以 50 就是 Similarity
Similarity = [(k,v/50) for k,v in Ultimate_candidate.items()]


# In[334]:


Similarity


# # 四、Top 10

# In[344]:


Top_10 = sorted(Similarity,key = lambda x: x[1],reverse = True)[:10]


# In[345]:


Top_10


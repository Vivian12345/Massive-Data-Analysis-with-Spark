#!/usr/bin/env python
# coding: utf-8

# # 1.把500input.txt以RDD的形式叫出來

# In[28]:


text_file = sc.textFile("500input.txt")


# # 2.以空格分開，每一行裡的元素再以逗號加以分開

# In[29]:


mm = text_file.map(lambda line: line.split(" ")).map(lambda word: word[0].split(",")) 


# # 3.為M與N矩陣分別造出以欄與列為key的RDD
# + mm_M.take(1) <br>
# [('0', ['M','0', '51'])]

# In[5]:


mm_M = mm.filter(lambda x: x[0] == "M").map(lambda x: (x.pop(2),x))


# In[6]:


mm_N = mm.filter(lambda x: x[0] == "N").map(lambda x: (x.pop(1),x))


# # 4.將兩個RDD以key聚合
# + mm_new.take(1) <br>
# [('4',
#   (<pyspark.resultiterable.ResultIterable at 0x1d2ce5de2b0>,
#    <pyspark.resultiterable.ResultIterable at 0x1d2ce602860>))]

# In[34]:


mm_new = mm_M.cogroup(mm_N)


# # 5.留下tuple的部分
# + mm_new.take(1) <br>
# [(<pyspark.resultiterable.ResultIterable at 0x1d2ce7cb080>,
#   <pyspark.resultiterable.ResultIterable at 0x1d2ce8f1978>)] <br>
# 前面一項會是一堆['M','0', '51']，後者則是['N',...]

# In[35]:


mm_new = mm_new.map(lambda x: x[1])


# # 6.Mapping:對於每一對tuple裡面的元素相互配對
# + mm_map.take(3) <br>
# [[(0, 0), 1840], [(0, 1), 1300], [(0, 2), 100]]

# In[9]:


def mapper1(x):
    list1 = []
    for j in x[0]:
        for k in x[1]:
            list1.append([(eval(j[1]),eval(k[1])),eval(j[2]+"*"+k[2])])
    return list1


# In[10]:


mm_map = mm_new.flatMap(mapper1)


# # 7.reduceByKey:將每個有一樣key的element加起來
# + ex. [(0,0),1840]和[(0,0),2445]會被加起來變成[(0,0),1840+2445]
# 
# # 8. sortByKey:並以key做排序
# # 9. persist:存在記憶體，加速後續的計算

# In[11]:


def reducer1(x,y):
    return x+y


# In[12]:


mm_sorted = mm_map.reduceByKey(reducer1,31).sortByKey().persist()


# # 10.造出Outputfile.txt準備寫入結果

# In[24]:


f = open("Outputfile.txt", "w")


# # 11. 將RDD從記憶體中取出並改成EX.(0,0,2638)的形式寫入txt

# In[25]:


for i in mm_sorted.unpersist().map(lambda x: (x[0][0],x[0][1],x[1])).collect():
    f.write(str(i[0])+","+str(i[1])+","+str(i[2])+"\n")


# In[26]:


f.close()


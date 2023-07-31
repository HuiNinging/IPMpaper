import numpy as np
import re
from collections import deque
from strsimpy.jaccard import Jaccard
from strsimpy.cosine import Cosine
from bert_bilstm_att_multiclass import predict,ModelConfig
model_config = ModelConfig()


test_comments = ['信息熵：表示某个概率系统的不确定程度，熵值越大，其系统的不确定程度越大。比如说，假设教师不了解学生的情况，那么提问时每个同学被提问的几率是相等的，对于这样的系统，我们很难预测那个同学会被提问到，这种系统的不确定性最大。'
                 '该系统的信息熵具有最大值。但如果教师对这个班的学生非常了解，并且习惯提问成绩较好的学生，那么该系统的不确定程度就会大大减少。'] #1


print('输入的评论：',test_comments)

#--------------------------------------------------------------------生成邻接矩阵-------------------------------------------------------------------------
def getdata(filename):
    linedata = open("ourknow/kg_new.txt", 'r')    #读取txt文件
    cnt = 0
    res = []
    n = len(open(r"ourknow/kg_new.txt").readlines())
    matrix = np.zeros((n, n))
    for line in linedata:
        linelist = [int(s) for s in line.split()] # 每一行根据分割后的结果存入列表
        res.append([])
        for x in linelist:
            res[cnt].append(x)
            matrix[max(res[cnt]) - 1][min(res[cnt]) - 1] = 1
            matrix[min(res[cnt]) - 1][max(res[cnt]) - 1] = 1
        cnt += 1
    return matrix

filename = 'ourknow/kg_new.txt'    # 存储的是知识点之间的关系
data = getdata(filename)

#--------------------------------------------------------------------广度搜索-------------------------------------------------------------------------

graph = data
def bfs(G,s):
    S,Q=set(),deque([s])
    result=[]
    while(Q):
        u=Q.popleft()
        if(u in S):
            continue
        S.add(u)
        for v in range(len(G)):
            if(G[u][v]==1):
                Q.append(v)
        result.append(u)
    return result

# 读取文件，获取节点和边---------------------------------------------------------
f = open("ourknow/kg_new.txt", "r")
sources = []
targets = []
while True:
    line = f.readline()
    if line:
        source = line.split('\t')[0]
        target = line.split('\t')[1]
        target = re.sub('([^\u4e00-\u9fa5\u0030-\u0039])', '', target)
        sources.append(source)
        targets.append(target)
    else:
        break
f.close()
m = len(list(set(sources)))

#--------------------------------------------------------------------生成特定主题路径-------------------------------------------------------------------------

import csv
with open('ourknow/theme_know.csv','r',encoding='gbk') as csvfile:
    reader1 = csv.DictReader(csvfile)
    column1 = [row['theme'] for row in reader1]
with open('ourknow/theme_know.csv','r',encoding='gbk') as csvfile:
    reader2 = csv.DictReader(csvfile)
    column2 = [row['know'] for row in reader2]

topic = predict(test_comments, model_config)  # 神经网络预测的主题
list1 = []
for i in range(len(column1)):
    if column1[i] == str(topic):
        list1.append(column2[i])

lujing=[]
for j in range(len(list1)):
    m = int(list1[j])
    lujing.append((bfs(graph,m)[0:5]))     # 广度搜索选择每条路径长度为5
# print('广度搜索主题对应的知识点路径：',lujing)
# --------------------------------------------------------------------求路径的相似度-------------------------------------------------------------------------
import pandas as pd
import jieba

# 加载停用词 ---------------------------------------------------------------------
def load_stopword():
    f_stop = open('data/hit_stopwords.txt', encoding='utf-8')
    sw = [line.strip() for line in f_stop]
    f_stop.close()
    return sw

# 中文分词并且去停用词--------------------------------------------------------------
def seg_word(sentence):
    sentence_seged = jieba.cut(sentence.strip(),HMM=True) # HMM参数可选可不选，默认为False
    stopwords = load_stopword()
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '/t':
                outstr += word
                outstr += " "
    return outstr

dataFile = 'data/knowledge_new.csv'
data = pd.read_csv(dataFile,encoding='gbk',header=None,index_col=0)
data1 = data.values.tolist()

fencilist=[]
for i in range(len(data1)):
    data11 = " ".join('%s' %id for id in data1[i])
    data11 = seg_word(data11)
    fencilist.append(data11)

# 每条路径下两两之间的相似度----------------------------------------------------------
cos = Cosine(1)
A = []
for i in range(len(lujing)):
    for j in range(5):
        for m in range(j+1,5):
           a = lujing[i][j]
           b = lujing[i][m]
           simj = cos.similarity(fencilist[a], fencilist[b])
           A.append(simj)

# 每条路径求和存储在列表里-------------------------------------------------------------
sum_every=[]
for n in range(len(A)//10):
    sum = 0
    for i in range(n*10,(n+1)*10):
        sum = sum + A[i]
    sum_every.append(sum)
# print('每条路径求和相似度：',sum_every)

# 找出求和最大的路径索引和路径-----------------------------------------------------------
max_lujing = max(sum_every)
max_suoyin = sum_every.index(max_lujing)
print('评分最高的路径',lujing[max_suoyin])

# 确定最大路径每个知识点的层级------------------------------------------------------------
layer_max = []
layer_know = []
for i in lujing[max_suoyin]:
    layer = data1[i][2]
    l_know = data1[i][0]
    layer_max.append(layer)
    layer_know.append(l_know)

print('学习路径推荐的知识点序列',layer_know)
# print('学习路径推荐的知识点对应层级：',layer_max)

# 输入的评论和知识作相似，确定评论层级-----------------------------------------------------
test_comments = " ".join('%s' %id for id in test_comments)
seg_comments = seg_word(test_comments)

input_sim = []
for i in range(len(fencilist)):
    input_cossim = cos.similarity(seg_comments, fencilist[i])
    input_sim.append(input_cossim)

max_inputsim = max(input_sim)
max_imputsim_suoyin = input_sim.index(max_inputsim)
# print('输入评论的知识层级:',data1[max_imputsim_suoyin][2])


# --------------------------------------------------------根据评论知识层级推荐用户------------------------------------------------------------------------------
# 原始数据按时间排序，提取内容列-----------------------------------------------------------
with open('data/datapart.csv','r',encoding='gbk') as csvfile:
    reader_data = csv.DictReader(csvfile)
    column_data = [row['content'] for row in reader_data]

# 关键短语的csv读取---------------------------------------------------------------------
with open('ourknow/keys/pyhanlp_datasum40.csv','r',encoding='utf-8') as csvfile:
    reader1 = csv.DictReader(csvfile)
    column = [row['key'] for row in reader1]
with open('ourknow/keys/pyhanlp_datasum40.csv', 'r', encoding='utf-8') as csvfile:
    reader2 = csv.DictReader(csvfile)
    id = [row['id'] for row in reader2]
with open('ourknow/keys/pyhanlp_datasum40.csv', 'r', encoding='utf-8') as csvfile:
    reader3 = csv.DictReader(csvfile)
    time = [row['time'] for row in reader3]

# 关键词的csv读取-----------------------------------------------------------------------
with open('ourknow/keys/keys_TFIDF50.csv','r',encoding='utf-8') as csvfile:
    reader11 = csv.DictReader(csvfile)
    column11 = [row['key'] for row in reader11]
with open('ourknow/keys/keys_TFIDF50.csv', 'r', encoding='utf-8') as csvfile:
    reader21 = csv.DictReader(csvfile)
    id21 = [row['id'] for row in reader21]
with open('ourknow/keys/keys_TFIDF50.csv', 'r', encoding='utf-8') as csvfile:
    reader31 = csv.DictReader(csvfile)
    time31 = [row['time'] for row in reader31]

# 清空历史文件数据------------------------------------------
his1 = open('ourknow/similarity/p_sim.csv', "r+")
his1.truncate()
his2 = open('ourknow/similarity/ascend_sim.csv', "r+")
his2.truncate()
# -----------------------------------------------

path = r'C:\Users\DELL\Desktop\recommend\ourknow\similarity\p_'
cos = Cosine(1)
jaccard = Jaccard(1)
filename = path + 'sim'
f = open(filename + '.csv', 'w', encoding='utf-8', newline='')
writer = csv.writer(f)
know_seg = fencilist[max_imputsim_suoyin]
for i in range(len(column)):
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []
    list6 = []
    list7 = []
    list1.append(id[i])
    list2.append(cos.similarity(know_seg, column[i]))
    list4.append(jaccard.similarity(know_seg, column[i]))
    list5.append(cos.similarity(know_seg, column11[i]))
    list6.append(0.25*(cos.similarity(know_seg, column[i])+jaccard.similarity(know_seg, column[i]))+0.5*cos.similarity(know_seg, column11[i]))
    list3.append(time[i])
    list7.append(column_data[i])
    rows = zip(list1,list3,list2,list4,list5,list6,list7)
    for row in rows:
        writer.writerow(row)
f.close()

# 排序处理：按照列值排序，升序排列，即时间从小到大---------------------------------------------------------------------------------
df = pd.read_csv('ourknow/similarity/p_sim.csv',header= None,names=['id','time','cosine','jaccard','words_cosine','sum_sim','content'])
data=df.sort_values(by="time", ascending=True)
data.to_csv('ourknow/similarity/ascend_sim.csv', index=False)

import pandas as pd
# 显示所有列(参数设置为None代表显示所有行，也可以自行设置数字)
pd.set_option('display.max_columns',None)
# 显示所有行
pd.set_option('display.max_rows',None)
# 设置数据的显示长度，默认为50
pd.set_option('max_colwidth',150)
# 禁止自动换行(设置为Flase不自动换行，True反之)
pd.set_option('expand_frame_repr', True)

# 选择符合相似度>0.75的行，包含用户id和帖子内容------------------------------------------------------------------------------------
arr = pd.read_csv('ourknow/similarity/ascend_sim.csv')
arr['sum_sim']=pd.to_numeric(arr['sum_sim'],errors='coerce')  # object转float
# 将time列设为索引
arr = arr.set_index("time")
arr1 = arr[arr['sum_sim']>0.75]

if len(arr1)>0:
    result = arr1.id.tolist()
    result1 = arr1.content.tolist()
    result2 = list(zip(result, result1))
    df1=pd.DataFrame(result2,columns=['id','content'])

    # 输出相似度大于0.75的用户的名字--------------------------------------------------------------------------------
    result_set = list(set((result)))
    print('相似用户列表：',result_set)
    # 将每个用户的帖子按照时间形成相似度列表，一个用户一行----------------------------------------------------------------
    import csv
    with open('ourknow/similarity/ascend_sim.csv','r',encoding='utf-8') as csvfile:
        reader_id = csv.DictReader(csvfile)
        column_id = [row['id'] for row in reader_id]
    with open('ourknow/similarity/ascend_sim.csv','r',encoding='utf-8') as csvfile:
        reader_sim = csv.DictReader(csvfile)
        column_sim = [row['sum_sim'] for row in reader_sim]

    father_list = []
    sim_list = []
    for i in range(len(result_set)):
        temp_list = []
        for j in range(len(column_id)):
            if result_set[i] == column_id[j]:
                if float(column_sim[j])>=0.7:
                    temp_list.append(column_sim[j])

        father_list.append(temp_list)

        # 按时间为每个相似用户的每个帖子赋值权重，求出每个用户的相似度------------------------------------------------------
        n = len(temp_list)
        sim = 0
        under = 0
        if n == 1:
            sim = float(temp_list[0])
        else:
            for m in range(n):
                up = m
                under = under + n -(m-1)
            sim = up/under
        sim_list.append(sim)
    # 找到相似度最大的数值及其索引----------------------------------------------------------------------
    max_sim = max(sim_list)
    max_simid = sim_list.index(max_sim)
    # 找到最大索引对应的用户名字，输出他的当前评论和下一条评论------------------------------------------------------------------------
    find_row=df1[df1['id']==str(result_set[max_simid])]
    print('推荐用户：',find_row['id'])
    print('推荐用户当前评论：',find_row['content'])
    result_df2 = arr.id.tolist()
    result1_df2 = arr.content.tolist()
    result2_df2 = list(zip(result_df2, result1_df2))
    df2=pd.DataFrame(result2_df2,columns=['id','content'])
    for n in range(int(max_simid+1),int(len(column_id))):
        if df2.iloc[n]['id'] == str(result_set[max_simid]):
            print('推荐用户的下一条评论:',df2.iloc[n]['content'])
            break








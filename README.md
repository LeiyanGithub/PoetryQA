# PoetryQA
a project of Intelligent question answering based on knowledge graph

关于诗词的知识图谱问答系统：
- 主要过程：
  - 实体识别：主要是因为诗名，诗内容以及诗状物等不常见，jieba分词不准确，所以将neo4j库中相应内容合并写入文件，作为自定义字典加入Jieba分词中
    - 实体有歧义，多个词性，比如李白，既是人名，也是诗中的词
    - 库中涉及实体太多，最好与jieba中求交
  - 问题理解：主要针对实体的属性、关系设计的问题
    - 设计问题对应的预料
    - 利用贝叶斯训练模型
  - 答案生成：针对问题设计的cypher模板，根据问题类别对应到答案
 
目前实际：
- 问题类型
![image](https://user-images.githubusercontent.com/45895439/113000566-865e7980-91a2-11eb-8c83-3fc4e69ce6fe.png)

- cypher查询语句
![image](https://user-images.githubusercontent.com/45895439/113001056-0258c180-91a3-11eb-8598-f37a27f3adcd.png)

- 测试样例
![image](https://user-images.githubusercontent.com/45895439/113001105-0f75b080-91a3-11eb-854d-a4ec79eda528.png)

- 答案生成
![image](https://user-images.githubusercontent.com/45895439/113000667-9fffc100-91a2-11eb-8261-8ac464f3c819.png)
![image](https://user-images.githubusercontent.com/45895439/113000736-af7f0a00-91a2-11eb-893f-0dc81b15ef8e.png)
![image](https://user-images.githubusercontent.com/45895439/113000791-bd348f80-91a2-11eb-9e72-870e48ad5341.png)
![image](https://user-images.githubusercontent.com/45895439/113000870-d0dff600-91a2-11eb-8052-dd70b1065bd5.png)
![image](https://user-images.githubusercontent.com/45895439/113000907-d9d0c780-91a2-11eb-84bb-2dd330971b42.png)


改进思想：
- 如何让问题与属性直接对应，实体与graph中的实体对应，但对于涉及关系类问题而言，需要思考
- 学习预训练，机器学习等方法

遇到的问题：
 - 	诗名相同，但内容不同，如何处理？
 - 	询问上下句的时候，可能是逗号对应的半句，也可能是句号对应一句
 - 	poetrything没有解释含义，是否合理
 -  诗mentionDynasty，这个关系适用于那些问题？与诗所属朝代有何不同

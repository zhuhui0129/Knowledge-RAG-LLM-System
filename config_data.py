
md5_path = "./md5.text"

DASHSCOPE_API_KEY = "sk-9cfe2919790c4bd88d10aa72493a5ec0"
# Chroma
collection_name="rag"
persist_directory="./chroma_db"

# spliter
chunk_size= 1000
chunk_overlap= 100
separators =["\n\n","\n",".","!","?","。","！","？"," ",""]

max_spliter_char_number= 1000  # 文本分割阈值

# 相似度K值
similarity_threshold =1     # 检索返回匹配的文档数量

embedding_model_name="text-embedding-v4"
chat_model_name="qwen3-max"
# chat_model_name="MiniMax-M2.5"

#
session_config = {
    "configurable": {
        "session_id": "user_001",
    }
}
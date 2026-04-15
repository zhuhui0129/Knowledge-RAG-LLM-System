import os
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from file_history_store import get_history
from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi

os.environ["DASHSCOPE_API_KEY"] = "sk-9cfe2919790c4bd88d10aa72493a5ec0"

def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)

    return prompt


class RagService(object):
    def __init__(self):

        # 从环境变量读取 DASHSCOPE_API_KEY
        dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
        if not dashscope_api_key:
            raise ValueError("请在环境变量中配置 DASHSCOPE_API_KEY")

        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(
                model=config.embedding_model_name,
                dashscope_api_key=dashscope_api_key
            )
        )

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "以我提供的已知参考资料为主，"
                 # "简洁和专业的回答用户问题。参考资料:{context}。"),                                                                                                                                                                
                 # ("system", "并且我提供用户的对话历史记录，如下："),                                                                                                                                                                       
                 #  MessagesPlaceholder("history"),  
                 "简洁和专业的回答用户问题。参考资料:{context}。"
                 "并且我提供用户的对话历史记录，如下：{history}"),
                ("user", "请回答用户提问：{input}")
            ]
        )

        self.chat_model = ChatTongyi(
            model=config.chat_model_name,
            dashscope_api_key=dashscope_api_key
        )

        # self.chat_model = MiniMax(api_key="sk-uH7co8iosbZs3NdAGVtITpCO6wD7cBa4eygEDggk5uttTYwS", model="MiniMax-M2.5")

        self.chain = self.__get_chain()

    def __get_chain(self):
        """获取最终的执行链"""

        retriever = self.vector_service.get_retriever()

        def format_document(docs: list[Document]):
            if not docs:
                return "无相关参考资料"

            formatted_str = ""
            for doc in docs:
                formatted_str += f"文档片段：{doc.page_content}\n文档元数据：{doc.metadata}\n\n"

            return formatted_str

        def format_for_retriever(value: dict)->str:

            return value["input"]

        def format_for_prompt_template(value):
            # {input, context, history}
            new_value = {}
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["context"]
            new_value["history"] = value["input"]["history"]
            return new_value


        chain = (
            {
                "input": RunnablePassthrough(),
                "context": RunnableLambda(format_for_retriever) | retriever | format_document
            }| RunnableLambda(format_for_prompt_template) |self.prompt_template | print_prompt |self.chat_model | StrOutputParser()
        )

        # 手动实现消息历史功能
        def add_history(chain, get_history_func):
            def wrapper(input, config=None):
                session_id = config.get("configurable", {}).get("session_id", "default") if config else "default"
                history_obj = get_history_func(session_id)
                history = history_obj.messages  # 获取消息列表
                if isinstance(input, str):
                    input = {"input": input}
                input["history"] = history
                return chain.invoke(input, config)
            return RunnableLambda(wrapper)

        conversation_chain = add_history(  # 增强的链
            chain,
            get_history,
        )

        return conversation_chain


if __name__ == '__main__':
    # session id 配置
    session_config ={
        "configurable":{
            "session_id":"user_001",
        }
    }
    res = RagService().chain.invoke({"input":"我之前问了什么"},session_config)
    print(res)


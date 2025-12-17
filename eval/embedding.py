from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, load_index_from_storage, StorageContext
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.indices.multi_modal import MultiModalVectorStoreIndex
from typing import List, Union, Optional
from llama_index.core.schema import ImageDocument, BaseNode, ImageNode
import os
from PIL import Image
from vl_embedding import VL_Embedding
import io
import hashlib
import warnings
import pickle
# 假设所有PPT页面为jpg格式存在某个文件夹下

# class VL_EmbeddingAdapter(BaseEmbedding):
#     def __init__(self, vl_embed: VL_Embedding):
#         self.vl_embed = vl_embed

#     def embed(self, nodes: List[BaseNode]) -> List[List[float]]:
#         """Embed document nodes."""
#         # 直接调用你定义的 __call__
#         nodes = self.vl_embed(nodes)
#         return [node.embedding for node in nodes]

#     def embed_query(self, query: str) -> List[float]:
#         """Embed a query string."""
#         return self.vl_embed.embed_text(query)[0]
def _hash_path(path: str) -> str:
    return hashlib.md5(path.encode("utf-8")).hexdigest()

def load_ppt_images_as_documents(folder_path: str, cache_path="./cache/image_doc_cache.pkl"):
    docs = []
    cache = {}

    # 如果存在缓存，先加载
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)

    updated = False
    for fname in os.listdir(folder_path):
        if fname.endswith(".png") or fname.endswith(".jpg"):
            fpath = os.path.join(folder_path, fname)
            key = _hash_path(fpath)

            if key in cache:
                docs.append(cache[key])
            else:
                image = Image.open(fpath)
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                img_bytes = buffer.getvalue()
                doc = ImageDocument(image=img_bytes, metadata={"file_path": fpath})
                docs.append(doc)
                cache[key] = doc
                updated = True

    # 写回缓存（只在有新文件时）
    if updated:
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)

    return docs

class RobustMultiModalIndexer:
    def __init__(self, embed_model, strict_validation=False):
        self.embed_model = embed_model
        self.strict_validation = strict_validation
        
    def _create_hybrid_nodes(self, docs: List[ImageDocument]) -> List[ImageNode]:
        """将ImageDocument转换为包含双重验证的ImageNode"""
        nodes = []
        for doc in docs:
            try:
                # 强制类型转换确保节点结构合规
                node = ImageNode(
                    image=doc.image,
                    text=doc.text or "",  # 确保不为None
                    metadata=doc.metadata,
                    excluded_embed_metadata_keys=["file_path"],  # 防止元数据干扰
                    excluded_llm_metadata_keys=["file_path"]
                )
                
                # 人工验证节点内容
                if not node.image and self.strict_validation:
                    raise ValueError("Empty image content")
                    
                nodes.append(node)
            except Exception as e:
                warnings.warn(f"Skipped invalid document {doc.metadata.get('file_path','')}: {str(e)}")
        return nodes

    def build_index(self, docs: List[ImageDocument], persist_dir: str, rebuild:bool = False) -> MultiModalVectorStoreIndex:
        """终极健壮的索引构建方法"""
        # 转换节点类型并过滤无效文档
        nodes = self._create_hybrid_nodes(docs)

         # 缓存加载逻辑
        if not rebuild and os.path.exists(persist_dir):
            print("🔍 尝试加载缓存索引...")
            try:
                storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
                cached_index = MultiModalVectorStoreIndex.from_vector_store(
                    storage_context.vector_store,
                    embed_model=self.embed_model,
                    _validate_nodes=False  # 禁用加载时的验证
                )
                print(f"✅ 加载缓存索引成功 (包含 {len(cached_index._index_struct.nodes)} 个节点)")
                return cached_index
            except Exception as e:
                print(f"❌ 缓存加载失败: {str(e)}. 将重建索引...")

        # 新建索引
        print("⚙️ 构建新索引...")
        storage_context = StorageContext.from_defaults(
            vector_store=SimpleVectorStore()
        )
        
        index = MultiModalVectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            image_embed_model=self.embed_model,
            image_field="image",
            text_field="text",
            is_image_to_text=False,
            show_progress=True,
            _validate_nodes=False
        )
        
        # 持久化时强制写入
        index.storage_context.persist(
            persist_dir=persist_dir,# 覆盖现有缓存
        )
        # print(f"✅ 索引构建完成 (包含 {len(index._index_struct.nodes)} 个节点)")
        return index
# # 4. 检索最相关的页面（图片）给定文本 query
def retrieve_topk(index, query, top_k=3):
    retriever = index.as_retriever(similarity_top_k=top_k)
    retrieved_nodes = retriever.retrieve(query)
    return retrieved_nodes

# 使用示例
if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"
    # 1. 初始化嵌入模型
    embed_model = VL_Embedding(
        model="vidore/colpali-v1.2",
        device="cuda:0",
        mode="image" 
    )
    query = "What are the economic implications of transitioning to a green economy in Southeast Asia?"
    # 2. 创建索引构建器
    index_builder =RobustMultiModalIndexer(embed_model)
    
    # 3. 加载图像文档
    image_docs = load_ppt_images_as_documents("/data2/home/yankai/ppt_crawler/data/trend/trend_images/bain_report_southeast_asias_green_economy_2025")
    # image_docs = index_builder._create_hybrid_nodes(image_docs)  # 确保转换为ImageNode
    
    # 4. 构建索引
    index = index_builder.build_index(image_docs, persist_dir="./cache/indexes/bain_report_index")
    top_img = retrieve_topk(index, query, top_k=10)
    print(f"Top {len(top_img)} images retrieved for query '{query}':")






# # 5. 提交给 LLM 生成答案
# def generate_answer(query, context_images, gpt_llm):
#     """
#     query: 用户问题
#     context_images: 经过检索的 ImageDocument 节点
#     gpt_llm: OpenAI GPT wrapper or LLM from llama_index
#     """
#     # 你可以选择附带 context metadata（比如文件名）给模型
#     context_str = "\n\n".join([f"Image page: {node.metadata.get('file_path', '')}" for node in context_images])
#     prompt = f"""
# You are answering a question based on visual slides (PPT pages). These are the most relevant slides:

# {context_str}

# Question: {query}

# Give a concise but informative answer.
# """
#     return gpt_llm.complete(prompt).text


# # 6. 整合示例
# if __name__ == "__main__":
#     folder = "./data/trend/trend_images/bain_report_southeast_asias_green_economy_2025"
#     query = "What strategies are proposed to decarbonize Southeast Asia’s economy?"

#     docs = load_ppt_images_as_documents(folder)
#     index = build_vector_index(docs, embed_model)

#     top_images = retrieve_topk(index, query, top_k=3)

#     # 你需要在这里配置 GPT LLM，例如：
#     # from llama_index.llms import OpenAI
#     # gpt_llm = OpenAI(model="gpt-4", api_key="...")
    
#     # answer = generate_answer(query, top_images, gpt_llm)
#     # print("Answer:", answer)
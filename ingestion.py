import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from fsspec import AbstractFileSystem
from pathlib import Path
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.readers.file import FlatReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import ImageNode, TextNode
from llama_index.core import SimpleDirectoryReader

from llms.vl_embedding import VL_Embedding
from typing import Any, Dict, List, Optional

# from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class PatchedFlatReader(FlatReader):
    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        fs = self._get_fs(file, fs)
        with fs.open(file, encoding="utf-8") as f:
            content = f.read()

        # 使用统一的字段名
        metadata = {
            "file_name": file.name,        # 改为 file_name（和 SimpleDirectoryReader 对齐）
            "extension": file.suffix
        }

        if extra_info:
            metadata = {**metadata, **extra_info}

        return [Document(text=content, metadata=metadata)]

class Ingestion:
    def __init__(self, dataset_dir,input_name='ppocr', input_type='ppocr',embed_model_name='BAAI/bge-m3'):
        self.dataset_dir = dataset_dir
        self.input_dir  = os.path.join(dataset_dir, "images/"+input_name)
        self.output_dir = os.path.join(dataset_dir, "nodes/"+input_name)
        self.output_file_format = 'document'
        self.chunk_size = 1024
        self.overlap_size = 0
        self.workers = 1
        self.reader = PatchedFlatReader()
        self.embed_model_name = embed_model_name
        self.type = input_type
        # colqwen/colpali/visrag(openbmb)
        if 'vidore' in embed_model_name or 'openbmb' in embed_model_name: 
            if self.type== 'img':
                self.reader = SimpleDirectoryReader(input_dir = self.input_dir)
                self.pipeline = IngestionPipeline(transformations=[
                    SimpleFileNodeParser(),
                    VL_Embedding(model=embed_model_name,mode='image')
                ])
                self.pipeline.disable_cache = True
            else:
                self.pipeline = IngestionPipeline(
                                    transformations=[
                                        SimpleFileNodeParser(),
                                        SentenceSplitter(
                                            include_metadata=True, include_prev_next_rel=True,
                                            chunk_size=self.chunk_size,
                                            chunk_overlap=self.overlap_size,
                                            separator=' ',       
                                            paragraph_separator='\n\n\n', secondary_chunking_regex='[^,.;。？！]+[,.;。？！]?'),
                                        VL_Embedding(model=embed_model_name,mode='text')
                                    ],
                                )
        else:
            self.pipeline = IngestionPipeline(
                                transformations=[
                                    SimpleFileNodeParser(),
                                    SentenceSplitter(
                                        include_metadata=True, include_prev_next_rel=True,
                                        chunk_size=self.chunk_size,
                                        chunk_overlap=self.overlap_size,
                                        separator=' ',       
                                        paragraph_separator='\n\n\n', secondary_chunking_regex='[^,.;。？！]+[,.;。？！]?'),
                                    HuggingFaceEmbedding(model_name=self.embed_model_name,trust_remote_code=True)
                                ],
                            )
    def close(self):
        import gc, torch

        # 清理 pipeline 内部的模型（如果有）
        if hasattr(self, 'pipeline'):
            try:
                if hasattr(self.pipeline, 'transformations'):
                    for t in self.pipeline.transformations:
                        # 释放 VL_Embedding 或 HuggingFaceEmbedding 的模型
                        if hasattr(t, 'model'):
                            del t.model
                        if hasattr(t, 'tokenizer'):
                            del t.tokenizer
                del self.pipeline
            except Exception as e:
                print(f"[Warning] 释放 pipeline 时出现异常: {e}")

        # 清理 reader
        if hasattr(self, 'reader'):
            try:
                del self.reader
            except Exception as e:
                print(f"[Warning] 释放 reader 时出现异常: {e}")

        # 强制回收内存与显存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


    def ingestion_example(self, input_file, output_file):
        # image
        if input_file.endswith('.jpg') or input_file.endswith('.png'):
            documents = self.reader.load_file(Path(input_file),self.reader.file_metadata,self.reader.file_extractor)
            # for doc in documents:
            #     print("→ Loaded document:", doc.metadata["file_name"])
            nodes = self.pipeline.run(documents=documents,num_workers=1, show_progress=False, cache_collection=False)
        elif self.type == "text_graph":
            documents = []
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for ent in data["entity"]:
                metadata = {
                    "file_name": input_file,
                    "entity_id": ent.get("id", ""),
                    "entity_type": ent.get("label", ""),
                }
                documents.append(TextNode(text=ent["text"], metadata=metadata))
            for rel in data["relation"]:
                metadata = {
                    "file_name": input_file,
                    "relation_id": rel.get("id" ,""),
                    "relation_type": rel.get("relationship", ""),
                    "source_id": rel.get("source_id", ""),
                    "target_id": rel.get("target_id", ""),
                    "confidence": rel.get("confidence", 0.0),
                    "keywords": rel.get("keywords", []),
                }
                documents.append(TextNode(text=rel.get("description", ""), metadata=metadata))
            nodes = self.pipeline.run(documents=documents,num_workers=1, show_progress=False, cache_collection=False)
        elif self.type == "graph":
            documents = []
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for ent in data["entity"]:
                content = ent.get("embedded_text", {}).get("content", "")
                if isinstance(content, list):
                    ent["text"] = ' '.join(content)
                elif isinstance(content, str):
                    ent["text"] = content
                else:
                    ent["text"] = str(content)
                metadata = {
                    "file_name": input_file,
                    "entity_id": ent.get("id", ""),
                    "entity_type": ent.get("label", ""),
                }
                documents.append(TextNode(text=ent["text"], metadata=metadata))
            for rel in data["relation"]:
                metadata = {
                    "file_name": input_file,
                    "relation_id": rel.get("id", ""),
                    "relation_type": rel.get("relationship", ""),
                    "source_id": rel.get("source_id", ""),
                    "target_id": rel.get("target_id", ""),
                    "confidence": rel.get("confidence", 0.0),
                    "keywords": rel.get("keywords", []),
                }
                documents.append(TextNode(text=rel.get("description", ""), metadata=metadata))
            
            nodes = self.pipeline.run(documents=documents,num_workers=1, show_progress=False, cache_collection=False)
        else: # txt
            documents=self.reader.load_data(Path(input_file))
            nodes = self.pipeline.run(documents=documents, show_progress=False, cache_collection=False)
        nodes_json = [node.to_dict() for node in nodes]
        if nodes_json is None or len(nodes_json) == 0:
            nodes_json = []
        with open(output_file, 'w') as json_file:
            json.dump(nodes_json, json_file, indent=2, ensure_ascii=False)
        return True
    
    def ingestion_multi_session(self):
        os.makedirs(self.output_dir, exist_ok=True)
        
        file_to_process = []
        for file in os.listdir(self.input_dir):
            file_prefix,_ = os.path.splitext(file)
            input_file = os.path.join(self.input_dir, file)
            output_file = os.path.join(self.output_dir, file_prefix) + '.node'
            if not os.path.exists(output_file):
                file_to_process.append((input_file, output_file))
        if self.workers == 1:
            for input_file, output_file in tqdm(file_to_process):
                self.ingestion_example(input_file, output_file)
        else:  
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                future_to_file = {executor.submit(self.ingestion_example, input_file, output_file): (input_file, output_file) for input_file, output_file in file_to_process}
                for future in tqdm(as_completed(future_to_file), total=len(file_to_process), desc='Processing files'):
                    result_type = future.result()
    


if __name__ == '__main__':
    root_path = './data'
    datasets = [ 'slideshare','tutorial']
    # datasets = ["ExampleDataset"]

    for dataset in datasets:
        dataset_dir = os.path.join(root_path, dataset)

        # select a embedding model
        # # ingestion = Ingestion(dataset_dir,input_prefix='img',output_prefix='colqwen_ingestion',embed_model_name='vidore/colqwen2-v1.0') # colqwen2
        # ingestion = Ingestion(dataset_dir,input_name='Internet_Trends_2012', input_type="img",
        #                       embed_model_name='vidore/colpali-v1.2') # colpali
        # # ingestion = Ingestion(dataset_dir,input_prefix='img',output_prefix='visrag_ingestion',embed_model_name='openbmb/VisRAG-Ret') # visrag
        # # ingestion = Ingestion(dataset_dir,input_prefix='ppocr',output_prefix='nv_ingestion',embed_model_name='nvidia/NV-Embed-v2') # nv-embed
        # # ingestion = Ingestion(dataset_dir,input_prefix='ppocr',output_prefix='bge_ingestion',embed_model_name='BAAI/bge-m3') # bge-m3
        # # ingestion = Ingestion(dataset_dir,input_prefix='graph',output_prefix='bge_graph',embed_model_name='BAAI/bge-m3')
        # # ingestion = Ingestion(dataset_dir,input_prefix='text_graph',output_prefix='bge_text_graph',embed_model_name='BAAI/bge-m3')
        # # run
        # ingestion.ingestion_multi_session()
        # ingestion.close()
        # print(f"Finished processing dataset: {dataset}")
        for name in tqdm(os.listdir(os.path.join(dataset_dir, "images")), desc=f"Processing dataset: {dataset}"):
            print(f"Processing dataset: {dataset}, input name: {name}")
            ingestion = Ingestion(dataset_dir,input_name=name, input_type="img",embed_model_name='vidore/colpali-v1.2') # bge-m3
            ingestion.ingestion_multi_session()
            ingestion.close()
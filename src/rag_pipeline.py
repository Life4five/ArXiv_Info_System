# rag_pipeline.py
import torch
import time
import gc
from typing import List, Dict
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


class ScienceRAG:
    def __init__(self, 
                 qdrant_host: str = "localhost", 
                 qdrant_port: int = 6333,
                 collection_name: str = "nlp2025_chunks", # Взял из твоего скрина
                 embed_model: str = "Qwen/Qwen3-Embedding-0.6B",
                 llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        
        self.collection_name = collection_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Инициализация RAG на устройстве: {self.device.upper()}")

        # 1. Подключение к базе знаний
        self.client = QdrantClient(qdrant_host, port=qdrant_port)
        
        # 2. Загрузка ретривера (Эмбеддинги) -> НА CPU (для экономии VRAM)
        print(f"📥 Загрузка ретривера: {embed_model}...")
        self.encoder = SentenceTransformer(embed_model, trust_remote_code=True, device="cpu")
        
        # Очистка памяти перед загрузкой LLM
        gc.collect()
        torch.cuda.empty_cache()

        # 3. Загрузка генератора (LLM) -> НА GPU (float16)
        print(f"🧠 Загрузка LLM: {llm_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model,
            torch_dtype=torch.float16, 
            attn_implementation="sdpa" 
        ).to(self.device)
        print("✅ Система готова к работе!\n")

    def _retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Внутренний метод: поиск векторов (Encoder на CPU)"""
        # encode автоматически использует девайс, указанный при инициализации (cpu)
        query_vector = self.encoder.encode(query, convert_to_numpy=True)
        
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )
        return [point.payload for point in search_result.points]

    def _format_context(self, chunks: list[dict]) -> str:
        formatted_text = ""
        for i, chunk in enumerate(chunks):
            title = chunk.get('title', 'Unknown Title')
            text = chunk.get('text', chunk.get('abstract', '')) 
            formatted_text += f"Document [{i+1}]\nTitle: {title}\nContent: {text}\n\n"
        return formatted_text

    def answer(self, query: str, top_k: int = 5) -> str:
        """Главный метод"""
        print(f"🔍 Ищу информацию по запросу: '{query}'...")
        retrieved_chunks = self._retrieve(query, top_k)
        
        if not retrieved_chunks:
            return "К сожалению, в базе знаний не найдено релевантной информации."

        context = self._format_context(retrieved_chunks)
        
        system_prompt = (
            "You are a helpful scientific assistant. "
            "Use the provided context to answer the user's question. "
            "Always cite the document titles used. "
            "If the context doesn't contain the answer, admit it."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]

        text_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Важно: inputs кидаем на GPU
        model_inputs = self.tokenizer([text_input], return_tensors="pt").to(self.device)

        print("✍️  Генерирую ответ...")
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=512,
                temperature=0.3,
                top_p=0.9,
                do_sample=True
            )

        response = self.tokenizer.batch_decode(
            [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)],
            skip_special_tokens=True
        )[0]
        
        return response
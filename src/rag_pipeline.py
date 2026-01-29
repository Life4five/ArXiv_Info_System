# rag_pipeline.py
import torch
import time
import gc
import logging
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScienceRAG:
    def __init__(self, 
                 qdrant_host: str = "localhost", 
                 qdrant_port: int = 6333,
                 collection_name: str = "nlp2025_chunks",
                 embed_model: str = "Qwen/Qwen3-Embedding-0.6B",
                 llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        
        self.collection_name = collection_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🚀 Initializing RAG on device: {self.device.upper()}")

        # 1. Подключение к Qdrant
        try:
            self.client = QdrantClient(qdrant_host, port=qdrant_port)
            # Проверка доступности коллекции
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if collection_name not in collection_names:
                raise ValueError(
                    f"Collection '{collection_name}' not found. "
                    f"Available: {collection_names}"
                )
            logger.info(f"✅ Connected to Qdrant collection: {collection_name}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Qdrant: {e}")
            raise
        
        # 2. Загрузка Encoder (на CPU для экономии VRAM)
        try:
            logger.info(f"📥 Loading retriever: {embed_model}...")
            self.encoder = SentenceTransformer(
                embed_model, 
                trust_remote_code=True, 
                device="cpu"
            )
            logger.info("✅ Retriever loaded on CPU")
        except Exception as e:
            logger.error(f"❌ Failed to load encoder: {e}")
            raise
        
        # Очистка памяти перед загрузкой LLM
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 3. Загрузка LLM (на GPU с float16)
        try:
            logger.info(f"🧠 Loading LLM: {llm_model}...")
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
            self.model = AutoModelForCausalLM.from_pretrained(
                llm_model,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                attn_implementation="sdpa" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            ).to(self.device)
            logger.info(f"✅ LLM loaded on {self.device.upper()}")
        except Exception as e:
            logger.error(f"❌ Failed to load LLM: {e}")
            raise
        
        logger.info("✅ RAG system ready!\n")

    def _retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Поиск релевантных документов в векторной базе.
        
        Args:
            query: Поисковый запрос
            top_k: Количество результатов
            
        Returns:
            Список payload'ов найденных документов
        """
        try:
            # Энкодинг запроса (на CPU)
            query_vector = self.encoder.encode(
                query, 
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            # Поиск в Qdrant
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector.tolist(),
                limit=top_k,
                with_payload=True,
                with_vectors=False
            )
            
            return [point.payload for point in search_result.points]
        
        except Exception as e:
            logger.error(f"❌ Retrieval error: {e}")
            return []

    def _format_context(self, chunks: List[Dict]) -> str:
        """
        Форматирование найденных документов в контекст для LLM.
        
        Args:
            chunks: Список payload'ов документов
            
        Returns:
            Форматированный текст контекста
        """
        if not chunks:
            return "No relevant documents found."
        
        formatted_text = ""
        for i, chunk in enumerate(chunks):
            # Безопасное извлечение полей
            title = chunk.get('title', 'Unknown Title')
            text = chunk.get('text') or chunk.get('abstract') or chunk.get('content', 'No content')
            
            formatted_text += (
                f"Document [{i+1}]\n"
                f"Title: {title}\n"
                f"Content: {text}\n\n"
            )
        
        return formatted_text

    def _extract_sources(self, chunks: List[Dict], max_text_len: int = 200) -> List[Dict[str, str]]:
        """
        Извлечение источников для фронтенда.
        
        Args:
            chunks: Список payload'ов
            max_text_len: Максимальная длина превью текста
            
        Returns:
            Список словарей с title и text
        """
        sources = []
        for chunk in chunks:
            title = chunk.get("title", "Unknown")
            text = chunk.get("text") or chunk.get("abstract") or chunk.get("content", "")
            
            # Обрезка текста для превью
            preview = text[:max_text_len] + "..." if len(text) > max_text_len else text
            
            sources.append({
                "title": title,
                "text": preview
            })
        
        return sources

    def answer(self, query: str, top_k: int = 5) -> Dict[str, any]:
        """
        Главный метод: поиск + генерация ответа.
        
        Args:
            query: Вопрос пользователя
            top_k: Количество источников
            
        Returns:
            Словарь с ключами 'answer' и 'sources'
        """
        logger.info(f"🔍 Processing query: '{query[:50]}...'")
        
        # 1. Retrieval
        retrieved_chunks = self._retrieve(query, top_k)
        
        if not retrieved_chunks:
            logger.warning("No documents found")
            return {
                "answer": "I couldn't find any relevant information in the knowledge base.",
                "sources": []
            }
        
        # 2. Подготовка источников для фронтенда
        sources = self._extract_sources(retrieved_chunks)
        
        # 3. Формирование контекста
        context = self._format_context(retrieved_chunks)
        
        # 4. Подготовка промпта
        system_prompt = (
            "You are a helpful scientific assistant with knowledge base of NLP arxiv paper for 2025 year. "
            "Use ONLY the provided context to answer the user's question."
            "If the context doesn't contain enough information, say so explicitly."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]

        # 5. Генерация ответа
        try:
            text_input = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            model_inputs = self.tokenizer(
                [text_input], 
                return_tensors="pt",
                truncation=True,
                max_length=4096
            ).to(self.device)

            logger.info("✍️  Generating answer...")
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=512,
                    temperature=0.3,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Декодирование только новых токенов
            response_text = self.tokenizer.batch_decode(
                [output_ids[len(input_ids):] 
                 for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)],
                skip_special_tokens=True
            )[0]
            
            # Очистка GPU памяти после генерации
            if self.device == "cuda":
                del model_inputs, generated_ids
                torch.cuda.empty_cache()
            
            logger.info("✅ Answer generated successfully")
            
            return {
                "answer": response_text.strip(),
                "sources": sources
            }
        
        except Exception as e:
            logger.error(f"❌ Generation error: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": sources
            }

    def cleanup(self):
        """Очистка ресурсов (опционально)"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'encoder'):
            del self.encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("🧹 Resources cleaned up")

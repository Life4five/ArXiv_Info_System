# ArXiv Info System: NLP Research Assistant (2025)

Вопросно-ответная RAG-система, построенная на корпусе научных статей категории cs.CL (Computation and Language) за 2025 год.

**Автор проекта:** [Демидов Константин](https://github.com/ConstDemi)  
**Руководитель проекта:** [Паточенко Евгений](https://github.com/evgpat)

**Статус проекта:** Proof of Concept

## Технологический стек
- LLM: `Qwen/Qwen2.5-1.5B-Instruct`
- Embeddings: `Qwen/Qwen3-Embedding-0.6B`
- Vector DB: `Qdrant (Dense vectors search)`
- Frameworks, libraries and tools: `arXiv API, Pandas, BeatifulSoup, Markdownify, Pyarrow (Parquet files), HF Datasets, Transformers, LangChain Text Splitters, Sentence Transformers, PyTorch, Streamlit, FastAPI, Uvicorn`
- Data Ops: `DVC, S3 storage`

## Архитектура и этапы пайплайна
1. Сбор метаданных и данных
    - Метаданные (`1metadata_parse.ipynb`): Автоматический сбор через arXiv API с обходом лимитов через помесячную нарезку. Фильтрация строго по `primary_category == 'cs.CL'`.
    - Загрузка HTML (`2html_parse.ipynb`): Асинхронное скачивание HTML-версий статей (`asyncio` + `httpx`). Реализована очистка от "битых" файлов и фильтрация по размеру (порог 20 Кб).

2. Препроцессинг и чанкование
    - Очистка (`3html_preprocess.ipynb`): Конвертация HTML в Markdown > удаление шума > сохранение формул в LaTeX формате.

    - Чанкование (`4chunking.ipynb`):
        1. Логическая нарезка по заголовкам (MarkdownHeaderTextSplitter)
        2. Рекурсивная нарезка (RecursiveCharacterTextSplitter)
        3. Удаление частей короче 50 символов

3. Индексация (`5indexing.ipynb`)
    - Создание эмбеддингов
    - Хранение векторов и их метаданных (payload) в коллекции Qdrant

4. Retrieval & Generation (`6RAG.ipynb`)
    - Retrieval: Поиск Top-k наиболее релевантных чанков по запросу пользователя
    - После работы эмбеддера очищаем кэш для освобождения ресурсов
    - Объявляем системные инструкции и генерируем ответ

## Структура репозитория
```
src/
├── 1metadata_parse.ipynb    # Парсинг метаданных с помощью API ArXiv
├── 2html_parse.ipynb        # Асинхронная скачка HTML статей
├── 3html_preprocess.ipynb   # HTML -> MD
├── 4chunking.ipynb          # Чанкование
├── 5embeddings.ipynb        # Генерация эмбеддингов и FAISS
├── 6db.ipynb                # Заливка обработнных данных в Qdrant
├── 7RAG.ipynb               # Тестируем RAG в Jupyter Notebook
├── 8get_random_chunks.ipynb #

data/                        # Папка для данных, которые участвуют во всём пайплайне от парсинга arxiv до заливки в Qdrant
├── metadata/                # CSV с метаданными статей
├── raw/                     # Сырые данные, требующие предобработки
└── processed/               # Обработанные данные, который заливаются в Qdrant
```


## Если хочется запустить проект локально:

### Предварительные требования
- Python 3.11+
- Git
- 20 GB свободного места на диске

### Шаг 1: Клонирование репозитория
```bash
git clone https://github.com/ConstDemi/ArXiv_Info_System.git
cd ArXiv_Info_System
```

### Шаг 2: Установка зависимостей
```bash
pip install -r requirements.txt
```

### Шаг 3: Настройка доступа к S3 хранилищу

1. Создайте файл `.dvc/config.local` на основе шаблона `.dvc/config.local.example`

2. Отредактируйте `.dvc/config.local`, заполнив переменные предоставленными credentials (доступы от S3 хранилища).

**Для членов комиссии:** credentials предоставляются отдельно.

### Шаг 4: Загрузка датасета
```bash
dvc pull
```

### Шаг 5: Восстанавливаем снапшот Qdrant коллеции

*TBD*

⚠️ **Внимание:** 
- Нотбуки 1-5 запускать не требуется - данные можно скачать командой `dvc pull`
- Скачивание датасета может занять до 30 минут (фикс будет 23.02.2026)

### Шаг 5: Запуск RAG системы

*TBD*
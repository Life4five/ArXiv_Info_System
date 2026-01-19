# ArXiv Info System: NLP Research Assistant (2025)

Вопросно-ответная RAG-система, построенная на корпусе научных статей категории cs.CL (Computation and Language) за 2025 год.

**Автор проекта:** [Демидов Константин](https://github.com/ConstDemi)  
**Руководитель проекта:** [Паточенко Евгений](https://github.com/evgpat)

**Статус проекта:** Proof of Concept

## Технологический стек
- LLM: `Qwen/Qwen2.5-1.5B-Instruct`
- Embeddings: `Qwen/Qwen3-Embedding-0.6B`
- Vector Search: `FAISS (IndexFlatIP)`
- Frameworks: `LangChain Splitters, SentenceTransformers, BeautifulSoup, Markdownify`
- Data Ops: `DVC, ArXiv API`

## Архитектура и этапы пайплайна
1. Сбор метаданных и данных
    - Метаданные (`1metadata_parse.ipynb`): Автоматический сбор через arXiv API с обходом лимитов через помесячную нарезку. Фильтрация строго по `primary_category == 'cs.CL'`.
    - Загрузка HTML (`2html_parse.ipynb`): Асинхронное скачивание HTML-версий статей (`asyncio` + `httpx`). Реализована очистка от "битых" файлов и фильтрация по размеру (порог 20 Кб).

2. Препроцессинг и чанкование
    - Очистка (`3html_preprocess.ipynb`): Конвертация HTML в Markdown. Удаление шума: разделы References, Acknowledgements, блоки авторов и ошибки парсинга. Сохранили формулы в LaTeX формате.

    - Чанкование (`4chunking.ipynb`):
        1. Логическая нарезка по заголовкам (MarkdownHeaderTextSplitter)
        2. Рекурсивная нарезка (RecursiveCharacterTextSplitter)
        3. Удаление частей короче 50 символов

3. Индексация (`5indexing.ipynb`)
    - Создание эмбеддингов
    - Хранение векторов в FAISS индексе с нормализацией для поиска по косинусному сходству

4. Retrieval & Generation (`6RAG.ipynb`)
    - Retrieval: Поиск Top-10 наиболее релевантных чанков по запросу пользователя
    - После работы эмбеддера очищаем кэш для освобождения ресурсов
    - Объявляем системные инструкции и генерируем ответ

## Структура репозитория
```
src/
├── 1metadata_parse.ipynb   # Парсинг метаданных с помощью API ArXiv
├── 2html_parse.ipynb       # Асинхронная скачка HTML статей
├── 3html_preprocess.ipynb  # HTML -> MD
├── 4chunking.ipynb         # Чанкование
├── 5indexing.ipynb         # Генерация эмбеддингов и FAISS
└── 6RAG.ipynb              # Поиск и генерация

data/
├── metadata/               # CSV с описанием статей
├── raw/html/               # Сырые HTML файлы статей
└── processed/              # Обработанные MD статьи, JSONL чанки и FAISS индекс
```


## Если хочется запустить проект локально:

### Предварительные требования
- Python 3.11+
- Git
- ~10 GB свободного места на диске

### Шаг 1: Клонирование репозитория
```bash
git clone https://github.com/ConstDemi/ArXiv_Info_System.git
cd ArXiv_Info_System
```

### Шаг 2: Установка зависимостей
```bash
pip install -r requirements.txt
```

### Шаг 3: Настройка доступа к хранилищу данных

1. Создайте файл `.dvc/config.local` на основе шаблона `.dvc/config.local.example`

2. Отредактируйте `.dvc/config.local`, заполнив переменные предоставленными credentials (доступы от S3 хранилища).

**Для членов комиссии:** credentials предоставляются отдельно.

### Шаг 4: Загрузка датасета
```bash
dvc pull
```

⚠️ **Внимание:** 
- Нотбуки 1-5 запускать не требуется - данные можно скачать командой `dvc pull`
- Скачивание датасета может занять до 30 минут (фикс будет 23.02.2026)

### Шаг 5: Запуск RAG системы

- После скачивания датасета командой `dvc pull` запустите `6RAG.ipynb`, написав свой запрос в переменную `QUERY`. В последней ячейке вы получите ответ от RAG.
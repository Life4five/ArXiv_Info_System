# ArXiv Info System

Вопросно-ответная RAG-система (RAG – Retrieval-Augmented Generation) по научным статьям с [arxiv.org](https://arxiv.org/).

**Автор проекта:** [Демидов Константин](https://github.com/ConstDemi)  
**Руководитель проекта:** [Паточенко Евгений](https://github.com/evgpat)


## Структура проекта

```text
info.ipynb               Паспорт проекта (описание целей, методологии, прогресса)
metadata_parse.ipynb     Сбор метаданных статей в датафрейм
data_parse.ipynb         Скачивание статей на основе созданного датафрейма
test_yandex_s3.ipynb     Код для проверки соединения с S3 хранилищем


data/                    рабочее хранилище данных проекта (создастся при запуске data_parse.ipynb)
└── raw/                 "сырые" (необработанные) статьи
    ├── tex/             архивы LaTeX-исходников статей (.tar.gz)
```

# ArXiv Info System

Вопросно-ответная RAG-система (RAG – Retrieval-Augmented Generation) по научным статьям с [arxiv.org](https://arxiv.org/).

**Автор проекта:** [Демидов Константин](https://github.com/ConstDemi)  
**Руководитель проекта:** [Паточенко Евгений](https://github.com/evgpat)


## Структура проекта

```text
info.ipynb               Паспорт проекта (описание целей, методологии, прогресса)
data_parse.ipynb         Парсинг статей arXiv

data/                    рабочее хранилище данных проекта (создастся при запуске data_parse.ipynb)
└── raw/                 "сырые" (необработанные) статьи
    ├── tex/             архивы LaTeX-исходников статей (.tar.gz)
    └── pdf/             PDF-версии статей
```

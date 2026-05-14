# prompt-filter

Программный модуль фильтрации prompt injection атак в запросах к LLM API

---

# Запуск
docker compose down
docker compose up -d

# Подготовка датасетов
docker compose exec filtering python -m app.core.training.download_datasets

# Тестирования
docker compose exec filtering pytest tests -v

# Эксперименты
docker compose --profile experiments run --rm experiments

# Обучение
docker compose exec filtering python -m app.core.training.train_sklearn
docker compose exec filtering python -m app.core.training.train_embedding

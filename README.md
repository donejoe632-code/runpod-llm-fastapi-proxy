# RunPod LLM Production Gateway (FastAPI)

Высокопроизводительный микросервис-шлюз для хостинга открытых моделей (DeepSeek-R1, Qwen-2.5, Llama-3) на облачных GPU RunPod (NVIDIA RTX 3090 / 4090 / A4000).

---

## Зачем это бизнесу?
При обработке больших массивов данных (парсинг документов, суммаризация чатов поддержки, скоринг лидов) стоимость коммерческих API (OpenAI GPT-4o) может достигать сотен долларов в месяц.
Хостинг открытых моделей на почасовых GPU RunPod с vLLM снижает себестоимость генерации **в 4–5 раз** при скорости от **45 до 85 токенов в секунду**.

## Архитектура решения:
```
[n8n / Telegram Bot / Web Backend]
       │
       ▼ (OpenAI-compatible REST: /v1/chat/completions)
[FastAPI Gateway & Auth Proxy]
       │
       ├─► [Rate Limiter & Token Counter]
       ├─► [Streaming SSE Transformer]
       │
       ▼ (vLLM Engine / Triton)
[RunPod Cloud GPU: NVIDIA RTX 4090 24GB]
       │
       ▼
[DeepSeek-R1 / Qwen-2.5-14B Weights]
```

## Функционал шлюза:
* **100% совместимость с OpenAI SDK**: достаточно изменить `base_url="http://your-server:8000/v1"` в любом коде.
* **Потоковая передача (SSE Streaming)**: моментальный первый токен (`TTFT < 250ms`).
* **Мониторинг метрик**: эндпоинты `/health` и `/metrics` с отслеживанием расхода памяти GPU и сэкономленного бюджета.
* **Авторизация**: защита по Bearer-токену.

## Быстрый запуск:
```bash
docker compose up -d
python test_proxy.py
```

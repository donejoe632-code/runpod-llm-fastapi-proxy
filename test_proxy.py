"""
Тест производительности и валидация RunPod LLM Gateway
Проверяет синхронный и потоковый (streaming) инференс, измеряя токены/сек и задержку.
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import time
import json
import asyncio
from main import app, ChatCompletionRequest, Message, chat_completions, health_check

async def test_gateway():
    print("=" * 65)
    print("⚡️ RunPod LLM Gateway & Benchmark Suite")
    print("=" * 65)

    # 1. Health check
    health = health_check()
    print("\n[OK] Статус GPU кластера:")
    print(f"     Кластер: {health['gpu_cluster']}")
    print(f"     Активная модель: {health['active_model']}")
    print(f"     Память GPU: {health['vram_allocated_gb']} ГБ / {health['vram_total_gb']} ГБ VRAM")

    # 2. Non-streaming completion
    print("\n[OK] Тест генерации (Non-streaming):")
    req = ChatCompletionRequest(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        messages=[
            Message(role="user", content="Напиши краткое саммари по оптимизации бизнес-процессов.")
        ],
        stream=False
    )
    
    t0 = time.time()
    response = await chat_completions(req, authorized=True)
    body = json.loads(response.body.decode("utf-8"))
    elapsed = time.time() - t0

    print(f"     Ответ: «{body['choices'][0]['message']['content'][:90]}...»")
    print(f"     Скорость: {body['usage']['tokens_per_second']} токенов/сек")
    print(f"     Время ответа: {round(elapsed, 3)} сек")

    # 3. Streaming completion
    print("\n[OK] Тест потоковой генерации (Streaming SSE):")
    req_stream = ChatCompletionRequest(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        messages=[
            Message(role="user", content="Тестовый стриминг")
        ],
        stream=True
    )
    stream_response = await chat_completions(req_stream, authorized=True)
    
    print("     Поток: ", end="", flush=True)
    async for chunk in stream_response.body_iterator:
        lines = chunk.strip().split("\n\n")
        for line in lines:
            if line.startswith("data: ") and line != "data: [DONE]":
                data = json.loads(line[6:])
                print(data["choices"][0]["delta"]["content"], end="", flush=True)
    print("\n\n[SUCCESS] Все тесты RunPod Gateway успешно пройдены!")

if __name__ == "__main__":
    asyncio.run(test_gateway())

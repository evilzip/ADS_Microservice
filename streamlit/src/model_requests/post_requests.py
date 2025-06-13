# импортируем библиотеку асинхронных запросов aiohttp
import aiohttp
# подключаем модуль для асинхронного кода asyncio
import asyncio


async def fetch_model_post(session: aiohttp.ClientSession, payload: dict, model_url: str):
    # открываем сеанс, отправляем запрос
    async with session.post(model_url, data=payload) as response:
        # возвращаем ответ от сервера
        return await response.json()


async def fetch_all(urls, payload):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_model_post(session=session, payload=payload, model_url=url) for url in urls]
        results = await asyncio.gather(*tasks)
        return results

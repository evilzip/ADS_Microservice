# импортируем библиотеку асинхронных запросов aiohttp
import aiohttp
# подключаем модуль для асинхронного кода asyncio
import asyncio


async def fetch_model_health(session: aiohttp.ClientSession, health_url: str):
    try:
        async with session.get(health_url) as response:
            response.raise_for_status()  # Raise exception for HTTP errors
            # status = await response.json()
            status = 'OK'
            # return data
    except aiohttp.ClientError as e:
        # print(f"Error fetching data from {health_url}: {e}")
        # status = f"Error fetching data from {health_url}: {e}"
        status = 'aiohttp.ClientError'
    except asyncio.TimeoutError:
        # print(f"Request to {health_url} timed out")
        # status = f"Request to {health_url} timed out"
        status = 'asyncio.TimeoutError'
    return status


async def fetch_all_health(health_urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_model_health(session=session, health_url=url) for url in health_urls]
        results = await asyncio.gather(*tasks)
        return results


# async def fetch_model_health(session: aiohttp.ClientSession, health_url: str):
#     try:
#         async with session.get(health_url) as response:
#             response.raise_for_status()  # Raise exception for HTTP errors
#             test = await response.json()
#             # return data
#     except aiohttp.ClientError as e:
#         # print(f"Error fetching data from {health_url}: {e}")
#         test = f"Error fetching data from {health_url}: {e}"
#     except asyncio.TimeoutError:
#         # print(f"Request to {health_url} timed out")
#         test = f"Request to {health_url} timed out"
#     return test
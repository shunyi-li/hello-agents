import requests
import os
from tavily import TavilyClient

from typing import Optional, Tuple


def _get_lat_lon(city: str) -> Optional[Tuple[float, float]]:
    """
    使用 Open-Meteo 的 Geocoding API 查询城市经纬度。
    返回 (latitude, longitude)，若未找到则返回 None。
    """
    geocode_url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {
        "name": city,
        "count": 1,
        "language": "zh",
        "format": "json"
    }
    try:
        resp = requests.get(geocode_url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if "results" in data and len(data["results"]) > 0:
            result = data["results"][0]
            return result["latitude"], result["longitude"]
        else:
            return None
    except Exception:
        return None


def get_weather(city: str) -> str:
    """
    通过 Open-Meteo 查询指定城市的当前天气。

    步骤：
    1. 先用 Geocoding API 获取城市经纬度
    2. 再用当前天气 API 获取实时天气

    注意：Open-Meteo 的“当前天气”需启用 `current_weather=true`
    """
    # 第一步：获取经纬度
    coords = _get_lat_lon(city)
    if not coords:
        return f"错误: 无法找到城市 '{city}' 的地理位置，请检查名称是否正确。"

    lat, lon = coords

    # 第二步：查询当前天气
    weather_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": True,
        "timezone": "auto"
    }

    try:
        response = requests.get(weather_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        current = data.get("current_weather")
        if not current:
            return f"错误: 未能获取 {city} 的当前天气数据。"

        temperature = current.get("temperature")
        weather_code = current.get("weathercode")

        # Open-Meteo 天气代码含义（简化版）
        WEATHER_CODES = {
            0: "晴天",
            1: "主要晴天",
            2: "局部多云",
            3: "多云",
            45: "雾",
            48: "冻雾",
            51: "小毛毛雨",
            53: "中毛毛雨",
            55: "大毛毛雨",
            61: "小雨",
            63: "中雨",
            65: "大雨",
            71: "小雪",
            73: "中雪",
            75: "大雪",
            95: "雷阵雨",
            96: "雷阵雨伴冰雹",
            99: "强雷阵雨伴冰雹"
        }

        desc = WEATHER_CODES.get(weather_code, f"天气代码{weather_code}")
        return f"{city}当前天气: {desc}，气温{temperature:.1f}摄氏度"

    except requests.exceptions.RequestException as e:
        return f"错误: 查询天气时遇到网络问题 - {e}"
    except Exception as e:
        return f"错误: 解析天气数据失败 - {e}"


def get_attraction(city: str, weather: str) -> str:
    """
    根据城市和天气，使用Tavily Search API搜索并返回优化后的景点推荐。
    """
    # 1. 从环境变量中读取API密钥
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "错误:未配置TAVILY_API_KEY环境变量。"

    # 2. 初始化Tavily客户端
    tavily = TavilyClient(api_key=api_key)

    # 3. 构造一个精确的查询
    query = f"'{city}' 在'{weather}'天气下最值得去的旅游景点推荐及理由"

    try:
        # 4. 调用API，include_answer=True会返回一个综合性的回答
        response = tavily.search(query=query, search_depth="basic", include_answer=True)

        # 5. Tavily返回的结果已经非常干净，可以直接使用
        # response['answer'] 是一个基于所有搜索结果的总结性回答
        if response.get("answer"):
            return response["answer"]

        # 如果没有综合性回答，则格式化原始结果
        formatted_results = []
        for result in response.get("results", []):
            formatted_results.append(f"- {result['title']}: {result['content']}")

        if not formatted_results:
            return "抱歉，没有找到相关的旅游景点推荐。"

        return "根据搜索，为您找到以下信息:\n" + "\n".join(formatted_results)

    except Exception as e:
        return f"错误:执行Tavily搜索时出现问题 - {e}"

# 将所有工具函数放入一个字典，方便后续调用
available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}

if __name__ == "__main__":
     weather = get_weather("杭州")
     print(weather)
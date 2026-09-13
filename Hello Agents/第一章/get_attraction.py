import os
from tavily import TavilyClient

def get_attraction(city: str, weather: str) -> str:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "错误：未配置TAVILY_API_KEY环境变量/。"
    tavily = TavilyClient(api_key = api_key)

    query = f"'{city}'在'{weather}'天气下最值得去的旅游景点推荐及利用"

    try:
        response = tavily.search(query = query, search_depth = "basic",include_answer = True)

        if response.get("answer"):
            return response["answer"]
        formatted_results = []
        for result in response.get("results", []):
            formatted_results.append(f"-{result['title']}:{result['content']}")

        if not formatted_results:
            return "抱歉，没有找到相关的旅游景点推荐"

        return "根据搜索，为您找到以下信息：\n" + "\n".join(formatted_results)

    except Exception as e:
        return f"错误：执行Tavily搜索时出现问题 - {e}"
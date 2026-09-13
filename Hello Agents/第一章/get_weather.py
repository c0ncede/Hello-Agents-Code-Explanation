import requests

def get_weather(city: str) -> str:
    """
    通过调用 wttr.in API 查询真实的天气信息
    """
    url = f"https://wttr.in/{city}?format=j1"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        current_condition = data['current_condition'][0]
        weather_desc = current_condition['weatherDesc'][0]['value']
        temp_c = current_condition['temp_C']

        return f"{city}当前天气：{weather_desc}, 气温{temp_c}摄氏度"

    except requests.exceptions.RequestException as e:
        return f"错误：查询天气遇到网络问题 - {e}"

    except (KeyError, IndexError) as e:
        return f"错误：解析天气数据失败，可能是城市名称无效 - {e}"
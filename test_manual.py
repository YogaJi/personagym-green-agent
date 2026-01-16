import requests
import json

# 这是你的裁判地址
url = "http://127.0.0.1:9009/tasks"

# 模拟一个比赛请求
payload = {
    "participants": {"pro": "http://fake-url", "con": "http://fake-url"},
    "config": {"topic": "AI is good"}
}

print("📨 正在发送请求给裁判...")
try:
    response = requests.post(url, json=payload)
    print(f"📡 状态码: {response.status_code}")
    print(f"📦 返回结果: {response.text}")
except Exception as e:
    print(f"❌ 连接失败: {e}")
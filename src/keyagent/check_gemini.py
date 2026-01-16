import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

# 1. 代理设置 (保持你之前的正确配置)
os.environ["http_proxy"] = "http://127.0.0.1:7890" 
os.environ["https_proxy"] = "http://127.0.0.1:7890"

print("🔍 再次连接 Google API 中...")

try:
    # 使用 v1beta，这是目前兼容性最好的版本
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
        http_options={'api_version': 'v1beta'} 
    )

    print("✅ 连接成功！正在拉取模型清单...\n")
    print("------------------------------------------------")
    
    # 直接列出所有模型，不进行属性过滤，防止报错
    for model in client.models.list():
        # 打印模型的“资源名称” (resource name)
        # 通常长这样: models/gemini-1.5-flash
        print(f"📦 发现模型: {model.name}")
        
    print("------------------------------------------------")
    print("🎉 列表获取完毕！")

except Exception as e:
    print(f"\n❌ 发生错误: {e}")
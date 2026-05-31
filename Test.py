from google import genai

client = genai.Client(api_key="AIzaSyDmPWPXZTKpwuTyDeUlMh5D8D4hiUgK9Io")

# 사용 가능한 모델 리스트 출력
for model in client.models.list():
    # 임베딩 기능을 지원하는 모델만 필터링
    if "embedContent" in model.supported_methods:
        print(f"모델 이름: {model.name}")
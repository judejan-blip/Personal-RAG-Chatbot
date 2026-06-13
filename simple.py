from openai import OpenAI

client = OpenAI(
    api_key="os.getenv()",
    base_url="https://openrouter.ai/api/v1"
)

completion = client.chat.completions.create(
    model="deepseek/deepseek-r1",
    messages=[
        {"role": "user", "content": "Give me 3 ideas for apps I could build with AI APIs"}
    ]
)

print(completion.choices[0].message.content)

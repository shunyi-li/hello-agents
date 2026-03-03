import agents

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # ---初始化 ---
    user_prompt = "你好，请帮我查询一下今天杭州的天气，然后根据天气推荐一个合适的旅游景点。"
    agents.run(user_prompt)
import sglang as sgl

@sgl.function
def multi_turn_question(s, question_1, question_2):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user(question_1)
    s += sgl.assistant(sgl.gen("answer_1", max_tokens=256))
    s += sgl.user(question_2)
    s += sgl.assistant(sgl.gen("answer_2", max_tokens=256))

def single():
    state = multi_turn_question.run(
        question_1="What is the capital of the United States?",
        question_2="List two local attractions.",
    )
    for m in state.messages():
        print(m["role"], ":", m["content"])
    print("\n-- answer_1 --\n", state["answer_1"])
    print("\n-- answer_2 --\n", state["answer_2"])

if __name__ == "__main__":
    backend = sgl.OpenAI(
        model_name="llama3.2:1b",
        base_url="http://127.0.0.1:11434/v1",
        api_key="EMPTY",
    )
    sgl.set_default_backend(backend)
    print("\n========== single ==========\n")
    single()